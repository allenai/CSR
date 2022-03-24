import os
from src.lightning.modules.sim_siam_module import SimSiamModule
from src.lightning.modules.moco2_module_old import MocoV2
from src.shared.utils import check_none_or_empty, load_lightning_inference, load_lightning_train
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from src.models.backbones import FeatureLearner, FeedForward
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, ConfusionMatrix
from torch.nn import Linear, modules


class LinearProbeModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        assert conf.num_classes != 0
        module = None
        self.encoder = None
        if not check_none_or_empty(conf.load_path) and os.path.exists(conf.load_path):
            if conf.module == 'MocoV2':
                module = None
                if self.conf.freeze:
                    assert False
                    module = load_lightning_inference(conf.load_path, MocoV2)
                else:
                    module = load_lightning_train(conf.load_path, MocoV2)
                self.encoder = module.encoder_q[0]

            elif conf.module == 'FeatureLearner':
                state_dict = torch.load(conf.load_path)['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder.resnet'):# and not k.startswith('encoder.resnet.fc'):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                module = MocoV2(
                    in_channels=conf.in_channels,
                    channel_width=conf.channel_width,
                    pretrained=conf.pretrained,
                    backbone_str=conf.backbone)
                module.encoder_q[0].load_state_dict(state_dict)
                if conf.freeze:
                    module.eval()
                    module.freeze()

                self.encoder = module.encoder_q[0]

                # self.encoder = FeatureLearner(in_channels=conf.in_channels)
                # self.encoder.load_state_dict(state_dict)

                # if conf.freeze:
                #     self.encoder.eval()

            else:
                raise ValueError('Unsupported module type')
        else:
            if conf.pretrained:
                print('[WARNING]: using ImageNet features')
            else:
                print('[WARNING]: using random features')
            module = MocoV2(
                in_channels=conf.in_channels,
                channel_width=conf.channel_width,
                pretrained=conf.pretrained,
                backbone_str=conf.backbone)
            if conf.freeze:
                module.eval()
                module.freeze()

            self.encoder = module.encoder_q[0]

        self.linear = Linear(512, conf.num_classes)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.test_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.val_misclass = {}

        self.save_hyperparameters()

    # will be used during inference
    def forward(self, x):
        return self.linear(self.encoder(x))

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x_dict, target = batch
        x = torch.cat((x_dict['image'], x_dict['mask_1'], x_dict['mask_2']), 1)
        pred = self(x)
        loss = F.cross_entropy(pred, target)

        # training metrics
        acc = self.train_acc(torch.argmax(pred, dim=1), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x_dict, target = batch
        x = torch.cat((x_dict['image'], x_dict['mask_1'], x_dict['mask_2']), 1)
        pred = self(x)
        loss = F.cross_entropy(pred, target)

        # training metrics
        flat_preds = torch.argmax(pred, dim=1)
        acc = self.val_acc(flat_preds, target)
        self.val_confmat(flat_preds, target)
        misclass_indicator = flat_preds != target
        indices = torch.arange(x.shape[0])

        self.val_misclass[batch_idx] = [indices[misclass_indicator],
                                        flat_preds[misclass_indicator], target[misclass_indicator]]
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x_dict, target = batch
        x = torch.cat((x_dict['image'], x_dict['mask_1'], x_dict['mask_2']), 1)
        pred = self(x)
        loss = F.cross_entropy(pred, target)

        # validation metrics
        flat_preds = torch.argmax(pred, dim=1)
        acc = self.test_acc(flat_preds, target)
        self.test_confmat(flat_preds, target)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = None
        if self.conf.optimizer == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.conf.lr,
                            momentum=0.9, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.conf.lr,
                             weight_decay=self.conf.weight_decay)
        else:
            raise NotImplemented('Optimizer not supported, need to add it.')

        scheduler = None
        if self.conf.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.conf.epochs, last_epoch=-1)
        else:
            raise NotImplemented('Scheduler not supported, need to add it.')

        lr_scheduler = {'scheduler': scheduler, 'monitor': 'val_acc'}

        return [optimizer], [lr_scheduler]

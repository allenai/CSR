import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from src.models.backbones import FeatureLearner
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, ConfusionMatrix


class ReceptacleModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.encoder = FeatureLearner(
            in_channels=conf.in_channels,
            channel_width=conf.channel_width,
            pretrained=conf.pretrained,
            num_classes=conf.num_classes,
            backbone_str=conf.backbone)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.test_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.val_misclass = {}
        self.save_hyperparameters()

    # will be used during inference
    def forward(self, x):
        return self.encoder(x)

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

"""
Adapted from: https://github.com/facebookresearch/moco

Original work is: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
This implementation is: Copyright (c) PyTorch Lightning, Inc. and its affiliates. All Rights Reserved

This implementation is licensed under Attribution-NonCommercial 4.0 International;
You may not use this file except in compliance with the License.

You may obtain a copy of the License from the LICENSE file present in this folder.
"""
from argparse import ArgumentParser
from typing import Union

import pytorch_lightning as pl
import torch
import torchvision
from torchmetrics.functional import accuracy
from src.models.backbones import FeatureLearner, FeedForward
from src.shared.utils import my_shuffle_evaluate
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class MocoV2(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_

    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.

    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:

        - `William Falcon <https://github.com/williamFalcon>`_

    Further modifications by:
        - `Samir Gadre <https://github.com/sagadre>`_

    """

    def __init__(
        self,
        in_channels=5,
        emb_dim: int = 512,
        num_negatives: int = 1024,  # 2048,  # 8192, #16384, 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 512,
        use_mlp: bool = True,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 512)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.emb_dim = emb_dim
        self.encoder_q, self.encoder_k = self.init_encoders()

        # if use_mlp:  # hack: brute-force replacement
        #     dim_mlp = self.hparams.emb_dim
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self):
        """
        Override to add your own encoders
        """

        backbone_q = FeatureLearner(
            in_channels=5,
            channel_width=64,
            pretrained=False,
            num_classes=self.hparams.emb_dim,
            backbone_str='resnet18')
        backbone_k = FeatureLearner(
            in_channels=5,
            channel_width=64,
            pretrained=False,
            num_classes=self.hparams.emb_dim,
            backbone_str='resnet18')

        projection_q = FeedForward(
            [self.emb_dim, self.emb_dim//2, self.emb_dim])
        projection_k = FeedForward(
            [self.emb_dim, self.emb_dim//2, self.emb_dim])

        encoder_q = nn.Sequential(backbone_q, projection_q)
        encoder_k = nn.Sequential(backbone_k, projection_k)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.hparams.num_negatives % batch_size != 0:
            assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, k, shuffle_q, hard_q, hard_k, re_k):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = k.shape[0]
        k_gather = concat_all_gather(k)
        hard_q_gather_this = None
        hard_k_gather_this = None
        re_k_gather_this = None
        shuffle_q_gather_this = None

        batch_size_all = k_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        if hard_q is not None:
            hard_q_gather_this = concat_all_gather(hard_q)[idx_this]
        if hard_k is not None:
            hard_k_gather_this = concat_all_gather(hard_k)[idx_this]
        if re_k is not None:
            re_k_gather_this = concat_all_gather(re_k)[idx_this]
        if shuffle_q is not None:
            shuffle_q_gather_this = concat_all_gather(shuffle_q)[idx_this]

        return k_gather[idx_this], shuffle_q_gather_this, hard_q_gather_this, hard_k_gather_this, re_k_gather_this, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self,
                img_q,
                img_k,
                shuffle_img_q=None,
                shuffle_img_q_idx=None,
                hard_q=None,
                hard_q_idx=None,
                hard_k=None,
                hard_k_idx=None,
                re_k=None,
                re_k_idx=None,
                update_queue=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update_queue:
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                img_k, shuffle_img_q, hard_q, hard_k, re_k, idx_unshuffle = self._batch_shuffle_ddp(
                    img_k, shuffle_img_q, hard_q, hard_k, re_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            shuffle_q = self.encoder_k(shuffle_img_q)
            shuffle_q = nn.functional.normalize(shuffle_q, dim=1)

            h_q = None
            # h_k = None
            # r_k = None
            if hard_q is not None:
                # h_q = self.encoder_k(hard_q.view(-1, 5, 224, 224))
                h_q = self.encoder_k(hard_q)
                h_q = nn.functional.normalize(h_q, dim=1)
                # h_q = h_q.view(k.shape[0], -1, self.emb_dim)
            # if hard_k is not None:
            #     h_k = self.encoder_k(hard_k.view(-1, 5, 224, 224))
            #     h_k = nn.functional.normalize(h_k, dim=1)
            #     h_k = h_k.view(k.shape[0], -1, self.emb_dim)
            # if re_k is not None:
            #     r_k = self.encoder_k(re_k.view(-1, 5, 224, 224))
            #     r_k = nn.functional.normalize(re_k, dim=1)
            #     r_k = r_k.view(k.shape[0], -1, self.emb_dim)

            # undo shuffle
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                shuffle_q = self._batch_unshuffle_ddp(shuffle_q, idx_unshuffle)

                if h_q is not None:
                    h_q = self._batch_unshuffle_ddp(h_q, idx_unshuffle)
                # if h_k is not None:
                #     h_k = self._batch_unshuffle_ddp(h_k, idx_unshuffle)
                # if r_k is not None:
                #     r_k = self._batch_unshuffle_ddp(r_k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_neg_shuffle = torch.einsum('nc,nc->n', [q, shuffle_q]).unsqueeze(-1)

        # l_neg_h_q = torch.einsum('nc,nkc->nk', [q, h_q])
        l_neg_h_q = torch.einsum('nc,nc->n', [q, h_q]).unsqueeze(-1)
        # l_neg_h_k = torch.einsum('nc,nkc->nk', [q, h_k])

        # logits: Nx(1+K) with temperature applied
        logits = torch.cat([l_pos, l_neg], dim=1) / \
            self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).type_as(logits)

        logits_h_q = torch.cat([l_pos, l_neg_h_q], dim=1) / \
            self.hparams.softmax_temperature
        logits_shuffle = torch.cat([l_pos, l_neg_shuffle], dim=1) / \
            self.hparams.softmax_temperature
        # logits_h_k = torch.cat([l_pos, l_neg_h_k], dim=1) / \
        #     self.hparams.softmax_temperature
        labels_h_q = torch.zeros(
            logits_h_q.shape[0], dtype=torch.long).type_as(logits)
        # labels_h_k = torch.zeros(
        #     logits_h_k.shape[0], dtype=torch.long).type_as(logits)
        labels_shuffle = torch.zeros(
            logits_shuffle.shape[0], dtype=torch.long).type_as(logits)

        # dequeue and enqueue
        if update_queue:
            self._dequeue_and_enqueue(k)

        # , logits_h_q, labels_h_q, logits_h_k, labels_h_k
        return logits, labels, logits_shuffle, labels_shuffle, logits_h_q, labels_h_q

    def training_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, False)

    def validation_epoch_end(self, outputs):

        def mean(res, key):
            # recursive mean for multilevel dicts
            return torch.stack([x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()

        log = {}
        for k in outputs[0]:
            log[k] = mean(outputs, k)

        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=100, last_epoch=-1)

        lr_scheduler = {'scheduler': scheduler, 'monitor': 'val_acc'}

        return [optimizer], [lr_scheduler]

    def _step_helper(self, batch, batch_idx, is_train):
        prefix = 'val'
        if is_train:
            prefix = 'train'

        q_dict, k_dict = batch
        img_q = torch.cat(
            (q_dict['image'], q_dict['mask_1'], q_dict['mask_2']), 1)
        img_k = torch.cat(
            (k_dict['image'], k_dict['mask_1'], k_dict['mask_2']), 1)
        shuffle_img_q = torch.cat(
            (q_dict['shuffle_image'], q_dict['shuffle_mask_1'], q_dict['shuffle_mask_2']), 1)
        in_frame_negatives = torch.cat(
            (q_dict['image'], q_dict['in_frame_negative_mask_1'], q_dict['in_frame_negative_mask_2']), 1)

        logits, labels, logits_shuffle, labels_shuffle, logits_h_q, labels_h_q = self(
            img_q=img_q,
            img_k=img_k,
            shuffle_img_q=shuffle_img_q,
            hard_q=in_frame_negatives,
            # hard_q_idx=q_dict['padding_in_frame_negatives'],
            # hard_k=k_dict['in_frame_negatives'],
            # hard_k_idx=k_dict['padding_in_frame_negatives'],
            update_queue=is_train)

        loss_con = F.cross_entropy(logits.float(), labels.long())

        dist = F.softmax(logits, 1).detach()
        target = labels.long().detach()

        acc1_con = accuracy(
            dist, target, top_k=1)
        acc5_con = accuracy(
            dist, target, top_k=5)

        # loss_h_q = F.cross_entropy(
        #     logits_h_q.float(), labels_h_q.long())
        # acc1_h_q, _ = precision_at_k(
        #     logits_h_q, labels_h_q, top_k=(1, 5))

        # loss_h_k = F.cross_entropy(
        #     logits_h_k.float(), labels_h_k.long())
        # acc1_h_k, _ = precision_at_k(
        #     logits_h_k, labels_h_k, top_k=(1, 5))

        shuffle_loss_mask = q_dict['has_shuffle_negative'] > 0.5
        in_frame_negative_loss_mask = q_dict['has_in_frame_negative'] > 0.5


        loss_shuffle = torch.tensor(13, device=self.device)
        loss_in_frame_negative = torch.tensor(13, device=self.device)

        if torch.any(shuffle_loss_mask):
            # NOTE: this term considers a different number of terms, should be ok as reduction='mean'
            #       by default. Might want to scale down this loss as less examples so probably trust
            #       the gradients less
            loss_shuffle = F.cross_entropy(
                logits_shuffle[shuffle_loss_mask].float(), labels_shuffle[shuffle_loss_mask].long())

        if torch.any(in_frame_negative_loss_mask):
            loss_in_frame_negative = F.cross_entropy(
                logits_h_q[in_frame_negative_loss_mask].float(), labels_h_q[in_frame_negative_loss_mask].long())
        # acc1_shuffle, _ = precision_at_k(
        #     logits_shuffle, labels_shuffle, top_k=(1, 5))

        loss_total = loss_con + 0.5 * loss_in_frame_negative # + loss_shuffle  # + loss_h_q + loss_h_k

        log = {f'{prefix}_loss_con': loss_con,
               f'{prefix}_acc1_con': acc1_con,
               f'{prefix}_acc5_con': acc5_con,
               f'{prefix}_loss_h_q': loss_in_frame_negative,
               #    f'{prefix}_acc1_h_q': acc1_h_q,
               #    f'{prefix}_loss_h_k': loss_h_k,
               #    f'{prefix}_acc1_h_k': acc1_h_k,
               f'{prefix}_loss_shuffle': loss_shuffle,
               #    f'{prefix}_acc1_shuffle': acc1_shuffle,
               f'{prefix}_loss': loss_total}

        if is_train:
            self.log_dict(log)

            return loss_total

        # case where we are taking a val step, return a dict for agg
        return log

# utils


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

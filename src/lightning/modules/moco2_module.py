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
        self.register_buffer("queue_edge", torch.randn(emb_dim, num_negatives))
        self.queue_edge = nn.functional.normalize(self.queue_edge, dim=0)

        self.register_buffer("queue_node", torch.randn(emb_dim, num_negatives))
        self.queue_node = nn.functional.normalize(self.queue_node, dim=0)

        self.register_buffer(
            "queue_edge_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "queue_node_ptr", torch.zeros(1, dtype=torch.long))

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
    def _dequeue_and_enqueue(self, k_node, k_edge):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            k_node = concat_all_gather(k_node)
            k_edge = concat_all_gather(k_edge)

        batch_size_node = k_node.shape[0]
        batch_size_edge = k_edge.shape[0]

        ptr_node = int(self.queue_node_ptr)
        ptr_edge = int(self.queue_edge_ptr)

        # if self.hparams.num_negatives % batch_size != 0:
        #     assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        if batch_size_node != 0:
            # replace the keys at ptr (dequeue and enqueue)
            if ptr_node + batch_size_node > self.hparams.num_negatives:
                margin = self.hparams.num_negatives - ptr_node
                self.queue_node[:, ptr_node:] = k_node.T[:, :margin]
                self.queue_node[:, :batch_size_node -
                                margin] = k_node.T[:, margin:batch_size_node]
            else:
                self.queue_node[:, ptr_node:ptr_node +
                                batch_size_node] = k_node.T
            # move pointer
            ptr_node = (
                ptr_node + batch_size_node) % self.hparams.num_negatives
            self.queue_node_ptr[0] = ptr_node

        if batch_size_edge != 0:
            if ptr_edge + batch_size_edge > self.hparams.num_negatives:
                margin = self.hparams.num_negatives - ptr_edge
                self.queue_edge[:, ptr_edge:] = k_edge.T[:, :margin]
                self.queue_edge[:, :batch_size_edge -
                                margin] = k_edge.T[:, margin:batch_size_edge]
            else:
                self.queue_edge[:, ptr_edge:ptr_edge +
                                batch_size_edge] = k_edge.T
            # move pointer
            ptr_edge = (
                ptr_edge + batch_size_edge) % self.hparams.num_negatives
            self.queue_edge_ptr[0] = ptr_edge

    @torch.no_grad()
    def _batch_shuffle_ddp(self, k):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = k.shape[0]
        k_gather = concat_all_gather(k)

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

        return k_gather[idx_this], idx_unshuffle

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
                is_self_feature,
                # queue_identifier,
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
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # split keys and queries into two streams for edge and self features
        k_node = k[is_self_feature]
        q_node = q[is_self_feature]
        k_edge = k[~is_self_feature]
        q_edge = q[~is_self_feature]

        logits_nodes, labels_nodes, logits_edges, labels_edges = None, None, None, None

        if k_node.shape[0] != 0:
            l_pos_nodes = torch.einsum(
                'nc,nc->n', [q_node, k_node]).unsqueeze(-1)
            l_neg_nodes = torch.einsum(
                'nc,ck->nk', [q_node, self.queue_node.clone().detach()])
            logits_nodes = torch.cat([l_pos_nodes, l_neg_nodes], dim=1) / \
                self.hparams.softmax_temperature
            labels_nodes = torch.zeros(
                logits_nodes.shape[0], dtype=torch.long).type_as(logits_nodes)

        if k_edge.shape[0] != 0:
            l_pos_edges = torch.einsum(
                'nc,nc->n', [q_edge, k_edge]).unsqueeze(-1)
            l_neg_edges = torch.einsum(
                'nc,ck->nk', [q_edge, self.queue_edge.clone().detach()])
            logits_edges = torch.cat([l_pos_edges, l_neg_edges], dim=1) / \
                self.hparams.softmax_temperature
            labels_edges = torch.zeros(
                logits_edges.shape[0], dtype=torch.long).type_as(logits_edges)

        # dequeue and enqueue
        if update_queue:
            self._dequeue_and_enqueue(k_node, k_edge)

        return logits_nodes, labels_nodes, logits_edges, labels_edges

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

        logits_node, labels_node, logits_edge, labels_edge = self(
            img_q=img_q,
            img_k=img_k,
            is_self_feature=q_dict['is_self_feature'],
            # queue_identifier=q_dict['queue_identifier'],
            update_queue=is_train)

        loss_node = torch.tensor(13., device=self.device)
        loss_edge = torch.tensor(13., device=self.device)
        acc1_node = torch.tensor(0., device=self.device)
        acc5_node = torch.tensor(0., device=self.device)
        acc1_edge = torch.tensor(0., device=self.device)
        acc5_edge = torch.tensor(0., device=self.device)

        if logits_node is not None:
            loss_node = F.cross_entropy(
                logits_node.float(), labels_node.long())

            dist = F.softmax(logits_node, 1).detach()
            target = labels_node.long().detach()

            acc1_node = accuracy(
                dist, target, top_k=1)
            acc5_node = accuracy(
                dist, target, top_k=5)

        if logits_edge is not None:
            loss_edge = F.cross_entropy(
                logits_edge.float(), labels_edge.long())
            dist = F.softmax(logits_edge, 1)
            target = labels_edge.long().detach()

            acc1_edge = accuracy(
                dist, target, top_k=1)
            acc5_edge = accuracy(
                dist, target, top_k=5)

        loss_total = loss_node + loss_edge

        log = {f'{prefix}_loss': loss_total,  # NOTE: DO NOT CHANGE THIS KEY IT IS USED FOR MONITOR
               f'{prefix}_loss_node': loss_node,
               f'{prefix}_loss_edge': loss_edge,
               f'{prefix}_acc1_node': acc1_node,
               f'{prefix}_acc5_node': acc5_node,
               f'{prefix}_acc1_edge': acc1_edge,
               f'{prefix}_acc5_edge': acc5_edge}

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

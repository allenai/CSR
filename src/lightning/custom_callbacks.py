from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.shared.utils import render_confusion_matrix
from torch.utils.data.dataloader import DataLoader


class ConfusionLogger(Callback):
    """ Custom callback to compute metrics at the end of each training epoch"""

    def __init__(self, class_names: List[str]):
        super().__init__()

        self.class_names = class_names

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.val_confmat.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        # collect validation data and ground truth labels from dataloader
        conf_matrix = pl_module.val_confmat.compute()
        conf_matrix = render_confusion_matrix(
            conf_matrix.cpu().numpy(), self.class_names)

        trainer.logger.experiment.log(
            {
                f"val_conf": wandb.Image(conf_matrix)
            })

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.test_confmat.reset()

    def on_test_epoch_end(self, trainer, pl_module):
        # collect validation data and ground truth labels from dataloader
        conf_matrix = pl_module.test_confmat.compute()
        conf_matrix = render_confusion_matrix(
            conf_matrix.cpu().numpy(), self.class_names)

        trainer.logger.experiment.log(
            {
                f"test_conf": wandb.Image(conf_matrix)
            })


class ContrastiveImagePredictionLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        if dataloader_idx == 0 and batch_idx == 0:
            # only log first batch in the dataloader
            self.__helper(trainer, batch, 'train')

    def on_val_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        if dataloader_idx == 0 and batch_idx == 0:
            # only log first batch in the dataloader
            self.__helper(trainer, batch, 'val')

    def __helper(
        self,
        trainer: 'pl.Trainer',
        batch: Any,
        prefix: str,
    ):
        # Bring the tensors to CPU
        query, key = batch

        q_img = query['image']
        q_mask_1 = query['mask_1']
        q_mask_2 = query['mask_2']
        q_rooms_ids = query['room_id']
        q_trajectory_ids = query['trajectory_id'].cpu().numpy()
        q_timesteps = query['timestep'].cpu().numpy()
        # has_shuffle_negatives = query['has_shuffle_negative'].cpu().numpy()

        k_img = key['image']
        k_mask_1 = key['mask_1']
        k_mask_2 = key['mask_2']
        k_rooms_ids = key['room_id']
        k_trajectory_ids = key['trajectory_id'].cpu().numpy()
        k_timesteps = key['timestep'].cpu().numpy()

        # s_img = query['shuffle_image']
        # s_mask_1 = query['shuffle_mask_1']
        # s_mask_2 = query['shuffle_mask_2']

        # Save the masks
        q_masks = [{
            "mask_1": {
                "mask_data": q_mask_1[i].squeeze().cpu().numpy(),
                "class_labels": {1: "mask1"}
            },
            "mask_2": {
                "mask_data": q_mask_2[i].squeeze().cpu().numpy()+1,
                "class_labels": {2: "mask2"}
            },
            "background": {
                "mask_data": (q_mask_1[i] + q_mask_2[i]).squeeze().cpu().numpy(),
                "class_labels": {0: "background"}
            }
        } for i in range(q_img.shape[0])]

        k_masks = [{
            "mask_1": {
                "mask_data": k_mask_1[i].squeeze().cpu().numpy(),
                "class_labels": {1: "mask1"}
            },
            "mask_2": {
                "mask_data": k_mask_2[i].squeeze().cpu().numpy()+1,
                "class_labels": {2: "mask2"}
            },
            "background": {
                "mask_data": (k_mask_1[i] + k_mask_2[i]).squeeze().cpu().numpy(),
                "class_labels": {0: "background"}
            }
        } for i in range(k_img.shape[0])]

        # s_masks = [{
        #     "mask_1": {
        #         "mask_data": s_mask_1[i].squeeze().cpu().numpy(),
        #         "class_labels": {1: "mask1"}
        #     },
        #     "mask_2": {
        #         "mask_data": s_mask_2[i].squeeze().cpu().numpy()+1,
        #         "class_labels": {2: "mask2"}
        #     },
        #     "background": {
        #         "mask_data": (s_mask_1[i] + s_mask_2[i]).squeeze().cpu().numpy(),
        #         "class_labels": {0: "background"}
        #     }
        # } for i in range(s_img.shape[0])]

        trainer.logger.experiment.log({
            f"{prefix}_queries": [wandb.Image(x, masks=mask, caption=f"room_id:{room_id}, trajectory_id:{trajectory_id}, timestep:{timestep}")
                                  for x, mask, room_id, trajectory_id, timestep in zip(q_img,
                                                                                       q_masks,
                                                                                       q_rooms_ids,
                                                                                       q_trajectory_ids,
                                                                                       q_timesteps)][:10],
            f"{prefix}_keys": [wandb.Image(x, masks=mask, caption=f"room_id:{room_id}, trajectory_id:{trajectory_id}, timestep:{timestep}")
                               for x, mask, room_id, trajectory_id, timestep in zip(k_img,
                                                                                    k_masks,
                                                                                    k_rooms_ids,
                                                                                    k_trajectory_ids,
                                                                                    k_timesteps)][:10],
        })


class ReceptacleImagePredictionLogger(Callback):
    def __init__(self, misclassification=True, every_n_val_epochs=5):
        super().__init__()
        self.every_n_val_epochs = every_n_val_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.current_epoch % self.every_n_val_epochs != self.every_n_val_epochs - 1:
            return

        masks = []
        images = None
        preds = None
        rooms_ids = None
        trajectory_ids = None
        timesteps = None
        targets = None

        # Bring the tensors to CPU
        for step, (val_input, val_label) in enumerate(trainer.datamodule.val_dataloader()):
            if step not in pl_module.val_misclass:
                break
            indices = pl_module.val_misclass[step][0]
            pred = pl_module.val_misclass[step][1]
            target = pl_module.val_misclass[step][2]

            image = val_input['image'][indices]
            mask_1 = val_input['mask_1'][indices]
            mask_2 = val_input['mask_2'][indices]
            rooms_id = val_input['room_id'][indices]
            trajectory_id = val_input['trajectory_id'][indices]
            timestep = val_input['timestep'][indices]

            # Save the masks
            masks += [{
                "mask_1": {
                    "mask_data": mask_1[i].squeeze().cpu().numpy(),
                    "class_labels": {1: "mask1"}
                },
                "mask_2": {
                    "mask_data": mask_2[i].squeeze().cpu().numpy()+1,
                    "class_labels": {2: "mask2"}
                },
                "background": {
                    "mask_data": (mask_1[i] + mask_2[i]).squeeze().cpu().numpy(),
                    "class_labels": {0: "background"}
                }
            } for i in range(indices.shape[0])]

            if images is not None:
                images = torch.cat((images, image), 0)
            else:
                images = image

            if targets is not None:
                targets = torch.cat((targets, target), 0)
            else:
                targets = target

            if preds is not None:
                preds = torch.cat((preds, pred), 0)
            else:
                preds = pred

            if rooms_ids is not None:
                rooms_ids = torch.cat((rooms_ids, rooms_id), 0)
            else:
                rooms_ids = rooms_id

            if trajectory_ids is not None:
                trajectory_ids = torch.cat((trajectory_ids, trajectory_id), 0)
            else:
                trajectory_ids = trajectory_id

            if timesteps is not None:
                timesteps = torch.cat((timesteps, timestep), 0)
            else:
                timesteps = timestep

        trainer.logger.experiment.log({
            "val_examples": [wandb.Image(x, masks=mask, caption=f"Pred:{pred}, Label:{y}, room_id:{room_id}, trajectory_id:{trajectory_id}, timestep:{timestep}")
                             for x, pred, y, mask, room_id, trajectory_id, timestep in zip(images,
                                                                                           preds,
                                                                                           targets,
                                                                                           masks,
                                                                                           rooms_ids,
                                                                                           trajectory_ids,
                                                                                           timesteps)]
        })

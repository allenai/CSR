import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from src.lightning.data_modules.contrastive_data_module import \
    ContrastiveDataModule
from src.lightning.modules.moco2_module import MocoV2
from src.lightning.custom_callbacks import ContrastiveImagePredictionLogger


class MocoV2Trainer(object):
    def __init__(self, conf):

        self.conf = conf
        seed_everything(self.conf.seed)

    def run(self):
        # Init our data pipeline
        dm = ContrastiveDataModule(self.conf.batch_size, self.conf.data_path, self.conf.train_object_representation)

        # To access the x_dataloader we need to call prepare_data and setup.
        dm.prepare_data()
        dm.setup()

        # Init our model
        model = None
        if self.conf.pretrain_path is not None and os.path.exists(self.conf.pretrain_path):
            model = MocoV2.load_from_checkpoint(self.conf.pretrain_path)
        else:
            model = MocoV2(num_negatives=self.conf.queue_size)

        wandb_logger = WandbLogger(project=self.conf.project_name,
                                   name=self.conf.experiment_name,
                                   job_type='train')

        # defining callbacks
        checkpoint_callback = ModelCheckpoint(dirpath=self.conf.checkpoint_path,
                                              filename='model/model-{epoch}-{val_loss:.2f}',
                                              verbose=True,
                                              monitor='val_loss',
                                              mode='min',
                                              every_n_val_epochs=5,
                                              save_top_k=-1)
        data_callback = ContrastiveImagePredictionLogger()
        learning_rate_callback = LearningRateMonitor(logging_interval='epoch')

        # set up the trainer
        trainer = pl.Trainer(max_epochs=self.conf.epochs,
                             check_val_every_n_epoch=5,
                             progress_bar_refresh_rate=self.conf.progress_bar_refresh_rate,
                             gpus=8,#self.conf.gpus,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback, learning_rate_callback, data_callback],
                             checkpoint_callback=True,
                             accelerator=self.conf.accelerator,
                             plugins=DDPPlugin(find_unused_parameters=False),
                             amp_level='O2',
                             precision=16)

        # Train the model
        trainer.fit(model, dm)

        # Evaluate the model on the held out test set
        # trainer.test()

        # Close wandb run
        wandb.finish()

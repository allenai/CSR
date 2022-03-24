import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from src.lightning.data_modules.receptacle_data_module import \
    ReceptacleDataModule
from src.lightning.modules.linear_probe_module import LinearProbeModule
from src.lightning.custom_callbacks import ConfusionLogger, ReceptacleImagePredictionLogger


class LinearProbeTrainer(object):
    def __init__(self, conf):

        self.conf = conf
        seed_everything(self.conf.seed)

    def run(self):
        # Init our data pipeline
        dm = ReceptacleDataModule(self.conf.batch_size, self.conf.data_path, self.conf.task)

        # To access the x_dataloader we need to call prepare_data and setup.
        dm.prepare_data()
        dm.setup()

        # Init our model
        model = LinearProbeModule(self.conf)

        wandb_logger = WandbLogger(project=self.conf.project_name,
                                   name=self.conf.experiment_name,
                                   job_type='train')

        # defining callbacks
        checkpoint_callback = ModelCheckpoint(dirpath=self.conf.checkpoint_path,
                                              filename='model/model-{epoch}-{val_acc:.2f}',
                                              verbose=True,
                                              monitor='val_loss',
                                              mode='min',
                                              every_n_val_epochs=5)
        learning_rate_callback = LearningRateMonitor(logging_interval='epoch')

        input_callback = ReceptacleImagePredictionLogger()

        confusion_callback = ConfusionLogger(self.conf.classes)

        # set up the trainer
        trainer = pl.Trainer(max_epochs=self.conf.epochs,
                             check_val_every_n_epoch=5,
                             progress_bar_refresh_rate=self.conf.progress_bar_refresh_rate,
                             gpus=self.conf.gpus,
                             logger=wandb_logger,
                             callbacks=[learning_rate_callback,
                                        checkpoint_callback,
                                        input_callback,
                                        confusion_callback],
                             checkpoint_callback=True)

        # Train the model
        trainer.fit(model, dm)

        # Evaluate the model on the held out test set
        trainer.test()

        # Close wandb run
        wandb.finish()

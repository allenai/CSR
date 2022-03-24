from src.shared.utils import worker_init_fn
import pytorch_lightning as pl
import src.dataloaders.augmentations as A
from src.dataloaders.contrastive_dataset import ContrastiveDataset
from src.dataloaders.contrastive_dataset_old import ContrastiveDatasetOld
from src.dataloaders.contrastive_dataset_object import ContrastiveDatasetObject
from src.shared.constants import (COLOR_JITTER_BRIGHTNESS,
                                  COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                                  COLOR_JITTER_SATURATION, DEFAULT_NUM_WORKERS,
                                  GRAYSCALE_PROBABILITY, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)
from src.shared.data_split import DataSplit
from torch.utils.data import DataLoader


class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, train_object_representation, use_old_dataset=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_object_representation = train_object_representation
        self.use_old_dataset = use_old_dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        D = None
        if self.use_old_dataset:
            D = ContrastiveDatasetOld
        else:
            if self.train_object_representation:
                D = ContrastiveDatasetObject
            else:
                D = ContrastiveDataset

        if stage == 'fit' or stage is None:

            self.train_set = D(
                self.data_dir, A.TrainTransform, DataSplit.TRAIN)
            self.val_set = D(
                self.data_dir, A.TestTransform, DataSplit.VAL)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = D(
                self.data_dir, A.TestTransform, DataSplit.TEST)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)

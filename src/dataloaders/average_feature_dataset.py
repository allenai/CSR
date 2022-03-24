from genericpath import exists
import json
import os
import torch

import torch
from src.shared.data_split import DataSplit
from torch.utils.data import Dataset


class AverageFeatureDataset(Dataset):

    def __init__(self, root_dir, data_split: DataSplit):
        # set the root directory
        self.feat_dir = os.path.join(root_dir, f'{data_split.name.lower()}_avg_feat')

        # set the dataset root, this is dependent on whether we are loading train or test data
        self.data_split = data_split

        subset_labels_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}_subset.json')

        self.data = None
        with open(subset_labels_filepath, 'r') as f:
            self.data = json.load(f)


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # randomly get a on example, under example, or unrelated example
        entry = self.data[idx]

        # get the label
        label = entry['receptacle']

        # load image
        room_id = entry['room_id']
        trajectory_id = entry['trajectory_id']
        first_object = entry['first_name']
        second_object = entry['second_name']


        data = torch.load(os.path.join(self.feat_dir, f'{first_object}_{second_object}_{room_id}_{trajectory_id}.pt'), map_location='cpu')


        # create dict and return
        return data, label

import itertools
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from src.shared.constants import CLASSES_TO_IGNORE, DATALOADER_BOX_FRAC_THRESHOLD, IMAGE_SIZE
from src.shared.data_split import DataSplit
from src.shared.utils import get_box
from torch.utils.data import Dataset
from torchvision import transforms as T


class ContrastiveDatasetObject(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit, balance_class=False, balance_instance=False, balance_self=False):
        # set the root directory
        self.root_dir = root_dir

        # set the dataset root, this is dependent on whether we are loading train or test data
        self.labels_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}.json')
        self.boxes_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}_boxes.json')

        # save the data augmentations that are to be applied to the images
        self.transform = transform
        assert self.transform is not None

        # load all of the ground truth actions into memory
        self.data_refs = None
        with open(self.labels_filepath) as f:
            self.data_refs = json.load(f)

        self.boxes = None
        with open(self.boxes_filepath) as f:
            self.boxes = json.load(f)

        # self.data = []
        self.nested_dict_node = {}
        self.__set_fixed_dataset(self.data_refs)
        # print(f'dataset size: {len(self.data)}')

    def __set_fixed_dataset(self, data_refs):
        nested_dict_node = {}

        for i, entry in enumerate(data_refs):
            name_1 = entry['first_name']
            name_2 = entry['second_name']
            room_id = entry['room_id']
            traj_id = entry['trajectory_id']

            c_1 = name_1.split('_')[0]
            c_2 = name_2.split('_')[0]

            # do some area filtering in the dataset
            top, bottom = self.boxes[f'{room_id}_{traj_id}_{entry["timestep"]}'][name_1]
            area = (bottom[0] - top[0]) * (bottom[1] - top[1])
            if area / (IMAGE_SIZE * IMAGE_SIZE) < DATALOADER_BOX_FRAC_THRESHOLD:
                continue
            top, bottom = self.boxes[f'{room_id}_{traj_id}_{entry["timestep"]}'][name_2]
            area = (bottom[0] - top[0]) * (bottom[1] - top[1])
            if area / (IMAGE_SIZE * IMAGE_SIZE) < DATALOADER_BOX_FRAC_THRESHOLD:
                continue

            hit_ignore_class = False
            for c in CLASSES_TO_IGNORE:
                if c in c_1 or c in c_2:
                    hit_ignore_class = True
                    break
            if hit_ignore_class:
                continue

            instance_key = None
            class_key = f'{c_1},{c_2}'
            if name_1 == name_2:
                instance_key = name_1

                if class_key in nested_dict_node:
                    if instance_key in nested_dict_node[class_key]:
                        nested_dict_node[class_key][instance_key].add(i)
                    else:
                        nested_dict_node[class_key][instance_key] = set([i])
                else:
                    nested_dict_node[class_key] = {}
                    nested_dict_node[class_key][instance_key] = set([i])

        for c in nested_dict_node:
            keys_to_del = []
            for inst in nested_dict_node[c]:
                if len(nested_dict_node[c][inst]) < 2:
                    keys_to_del.append(inst)
                else:
                    nested_dict_node[c][inst] = list(
                        itertools.permutations(nested_dict_node[c][inst], 2))

            for k in keys_to_del:
                del nested_dict_node[c][k]

        keys_to_del = []
        for c in nested_dict_node:
            if len(nested_dict_node[c]) == 0:
                keys_to_del.append(c)

        for k in keys_to_del:
            del nested_dict_node[k]

        self.nested_dict_node = nested_dict_node

    def __len__(self):

        return 800000

    def __getitem__(self, idx):
        data_bank = self.nested_dict_node

        # sample class
        sampled_class = random.choice(list(data_bank.keys()))

        # sample instance
        sampled_instance = random.choice(list(data_bank[sampled_class].keys()))

        # sample pair
        key1, key2 = random.choice(
            list(data_bank[sampled_class][sampled_instance]))

        lookup_pair = [self.data_refs[key1], self.data_refs[key2]]
        data_pair = []

        for i in range(2):
            entry = lookup_pair[i]

            # load image
            room_id = entry['room_id']
            trajectory_id = entry['trajectory_id']
            timestep = entry['timestep']
            first_object = entry['first_name']
            second_object = entry['second_name']
            im = Image.open(os.path.join(
                self.root_dir, f'{room_id}_{trajectory_id}_{timestep}.png'))

            # load masks
            m1 = get_box(
                self.boxes[f'{room_id}_{trajectory_id}_{timestep}'][first_object])
            m2 = get_box(
                self.boxes[f'{room_id}_{trajectory_id}_{timestep}'][second_object])

            assert first_object == second_object
            queue_identifier = abs(hash(first_object)) % (10 ** 8)


            data = {'mask_1': m1, 'mask_2': m2, 'image': im, 'room_id': room_id,
                    'trajectory_id': trajectory_id, 'timestep': timestep,
                    'is_self_feature': True,
                    'queue_identifier': queue_identifier}

            self.transform(data)

            data_pair.append(data)

        # create dict and return
        return data_pair[0], data_pair[1]

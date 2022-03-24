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


class ContrastiveDatasetOld(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit, max_in_frame_negatives=2, relational=False, query_negatives=False, key_negatives=False, shuffle_pickup_only=True):
        # set the root directory
        self.root_dir = root_dir
        self.shuffle_pickup_only = shuffle_pickup_only
        self.toTensor = T.ToTensor()
        self.relational = relational
        # we are going to mine hard negatives from the same frame (different relationships)
        # becase the number of such examples can be really large and differ for different
        # frames we set max_in_frame_negatives, which will cap the number of these kinds
        # of negatives sampled for each input data point
        self.max_in_frame_negatives = max_in_frame_negatives

        # set the dataset root, this is dependent on whether we are loading train or test data
        self.labels_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}.json')
        self.boxes_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}_boxes.json')
        self.boxes_shuffle_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}_boxes_shuffle.json')

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

        self.boxes_shuffle = None
        with open(self.boxes_shuffle_filepath) as f:
            self.boxes_shuffle = json.load(f)

        self.relations = []
        self.data = []
        self.edge_histogram = {}
        self.class_histogram = {}

        c = 0
        s = set()
        for r in self.data_refs:
            s.add(r['first_name'])
            s.add(r['second_name'])

        self.__set_fixed_dataset(self.data_refs, self.relational)

        if self.relational:
            print(f'dataset size: {len(self.relations)}')
        else:
            print(f'dataset size: {len(self.data)}')

    def __set_fixed_dataset(self, data_refs, relational):
        positives = {}
        self.class_histogram = {}

        for i, entry in enumerate(data_refs):
            part_1 = entry['first_name']
            part_2 = entry['second_name']
            room_id = entry['room_id']
            traj_id = entry['trajectory_id']
            timestep = entry['timestep']
            has_shuffle_negatives = entry['has_shuffle_negatives']

            if part_1 == part_2:
                assert not has_shuffle_negatives

            if has_shuffle_negatives and self.shuffle_pickup_only:
                with open(f'{self.root_dir}/{room_id}_{traj_id}_{timestep}_shuffle.txt', 'r') as f:
                    contents = f.readline()
                    if 'open' in contents:
                        has_shuffle_negatives = False
                        entry['has_shuffle_negatives'] = False

            c_1 = part_1.split('_')[0]
            c_2 = part_2.split('_')[0]

            # do some area filtering in the dataset
            top, bottom = self.boxes[f'{room_id}_{traj_id}_{entry["timestep"]}'][part_1]
            area = (bottom[0] - top[0]) * (bottom[1] - top[1])
            if area / (IMAGE_SIZE * IMAGE_SIZE) < DATALOADER_BOX_FRAC_THRESHOLD:
                continue
            top, bottom = self.boxes[f'{room_id}_{traj_id}_{entry["timestep"]}'][part_2]
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

            if c_1 in self.class_histogram:
                self.class_histogram[c_1] += 1
            else:
                self.class_histogram[c_1] = 1
            if c_2 in self.class_histogram:
                self.class_histogram[c_2] += 1
            else:
                self.class_histogram[c_2] = 1

            key = f'{part_1},{part_2},{room_id},{traj_id}'
            class_key = f'{c_1},{c_2}'
            if key in positives:
                positives[key].append((i, has_shuffle_negatives, class_key))
            else:
                positives[key] = [(i, has_shuffle_negatives, class_key)]

        self.relations = []
        self.data = []
        self.edge_histogram = {}

        for p in positives:
            # if len(positives[p]) == 1 and not positives[p][0][1]:
            #     continue
            if len(positives[p]) < 2:
                continue
            w_sh = [e[0] for e in positives[p] if e[1]]
            wo_sh = [e[0] for e in positives[p] if not e[1]]

            # prioritize samples with rearangement negatives
            positive_pairs = list(itertools.product(w_sh, w_sh))
            np.random.shuffle(positive_pairs)

            positive_negative_pairs = list(itertools.product(w_sh, wo_sh))
            np.random.shuffle(positive_negative_pairs)

            negative_positive_pairs = list(itertools.product(wo_sh, w_sh))
            np.random.shuffle(negative_positive_pairs)

            negative_pairs = list(itertools.product(wo_sh, wo_sh))
            np.random.shuffle(negative_pairs)

            prelim = positive_pairs + positive_negative_pairs + \
                negative_positive_pairs + negative_pairs
            tmp = []
            for t in prelim:
                if t[0] != t[1]:
                    tmp.append(t)

            assert len(tmp) == len(set(tmp))

            if relational:
                self.relations.append(tmp)
            else:
                s = tmp
                if positives[p][0][2] in self.edge_histogram:
                    self.edge_histogram[positives[p][0][2]] += len(s)
                else:
                    self.edge_histogram[positives[p][0][2]] = len(s)

                self.data += s

    def __len__(self):

        if self.relational:
            return len(self.relations)

        return len(self.data)

    def __getitem__(self, idx):
        # randomly get a on example, under example, or unrelated example
        key1, key2 = None, None
        if self.relational:
            key1, key2 = random.choice(self.relations[idx])
        else:
            key1, key2 = self.data[idx]
        lookup_pair = [self.data_refs[key1], self.data_refs[key2]]
        data_pair = []

        for i in range(2):
            entry = lookup_pair[0]

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

            data = {'mask_1': m1, 'mask_2': m2, 'image': im, 'room_id': room_id,
                    'trajectory_id': trajectory_id, 'timestep': timestep, 'self': int(first_object==second_object)}

            if entry['has_shuffle_negatives']:
                data['shuffle_image'] = Image.open(os.path.join(
                    self.root_dir, f'{room_id}_{trajectory_id}_{timestep}_shuffle.png'))
                data['shuffle_mask_1'] = get_box(
                    self.boxes_shuffle[f'{room_id}_{trajectory_id}_{timestep}'][first_object])
                data['shuffle_mask_2'] = get_box(
                    self.boxes_shuffle[f'{room_id}_{trajectory_id}_{timestep}'][second_object])
                data['has_shuffle_negative'] = 1
            else:
                data['shuffle_image'] = im.copy()
                data['shuffle_mask_1'] = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
                data['shuffle_mask_2'] = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
                data['has_shuffle_negative'] = 0

            if len(entry['in_frame_negatives']) != 0:
                negative_choice = entry['in_frame_negatives'][np.random.randint(
                    0, high=len(entry['in_frame_negatives']))]
                data['in_frame_negative_mask_1'] = get_box(
                    self.boxes[f'{room_id}_{trajectory_id}_{timestep}'][negative_choice[0]])
                data['in_frame_negative_mask_2'] = get_box(
                    self.boxes[f'{room_id}_{trajectory_id}_{timestep}'][negative_choice[1]])
                data['has_in_frame_negative'] = 1
            else:
                data['in_frame_negative_mask_1'] = torch.zeros(
                    1, IMAGE_SIZE, IMAGE_SIZE)
                data['in_frame_negative_mask_2'] = torch.zeros(
                    1, IMAGE_SIZE, IMAGE_SIZE)
                data['has_in_frame_negative'] = 0


            self.transform(data)

            data_pair.append(data)

        # create dict and return
        return data_pair[0], data_pair[1]

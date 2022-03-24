import json
import os
import random
from src.shared.utils import get_box
from src.shared.constants import CLASSES_TO_IGNORE

from PIL import Image
from src.shared.data_split import DataSplit
from torch.utils.data import Dataset


class ReceptacleDataset(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit, dump_data_subset: bool = False, load_data_subset: bool = True, balance: bool = True, task: str = 'receptacle'):
        # set the root directory
        self.root_dir = root_dir
        self.task = task
        assert self.task in {'receptacle', 'sibling', 'combined'}

        self.dump_data_subset = dump_data_subset
        self.load_data_subset = load_data_subset
        self.balance = balance

        # set the dataset root, this is dependent on whether we are loading train or test data
        self.labels_filepath = None
        self.data_split = data_split

        sufix = ''
        if load_data_subset:
            sufix = '_subset'

        self.labels_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}{sufix}.json')
        self.boxes_filepath = os.path.join(
            root_dir, f'{data_split.name.lower()}_boxes.json')

        # save the data augmentations that are to be applied to the images
        self.transform = transform

        # load all of the ground truth actions into memory
        data_references_raw = None
        with open(self.labels_filepath, 'r') as f:
            data_references_raw = json.load(f)
        self.boxes = None
        with open(self.boxes_filepath) as f:
            self.boxes = json.load(f)

        self.data = []
        self.__set_fixed_dataset(data_references_raw)

    def __set_fixed_dataset(self, data_references_raw):
        data_references = {'on': [], 'under': [], 'unrelated': [], 'sibling': []}
        for entry in data_references_raw:
            part_1 = entry['first_name']
            part_2 = entry['second_name']

            hit_ignore_class = False
            for c in CLASSES_TO_IGNORE:
                if c in part_1 or c in part_2:
                    hit_ignore_class = True
                    break
            if hit_ignore_class:
                continue

            if entry['receptacle'] == 1:
                assert(entry['receptacle_sibling'] != 1)
                data_references['on'].append(entry)
            elif entry['receptacle'] == 2:
                assert(entry['receptacle_sibling'] != 1)
                data_references['under'].append(entry)
            elif (entry['receptacle_sibling'] == 1) and (entry['first_name'] != entry['second_name']):
                data_references['sibling'].append(entry)
            elif entry['first_name'] != entry['second_name']:
                assert(entry['receptacle_sibling'] != 1)
                data_references['unrelated'].append(entry)

        assert len(data_references['on']) == len(data_references['under'])

        samples_per_category = min(min(len(data_references['on']), len(data_references['unrelated'])), len(data_references['sibling']))

        # balance the dataset with random unrelated samples

        self.data += random.sample(
            data_references['unrelated'], samples_per_category)
        if self.task == 'sibling' or self.task == 'combined':
            self.data += random.sample(
                data_references['sibling'], samples_per_category)
        if self.task == 'receptacle' or self.task == 'combined':
            self.data += random.sample(
                data_references['on'], samples_per_category)
            self.data += random.sample(
                data_references['under'], samples_per_category)

        if self.dump_data_subset:
            with open(os.path.join(self.root_dir, f'{self.data_split.name.lower()}_subset.json'), 'w') as f:
                json.dump(self.data, f, indent=4)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # randomly get a on example, under example, or unrelated example
        entry = self.data[idx]

        # get the label
        label = None

        if self.task == 'receptacle':
            label = entry['receptacle']
        elif self.task == 'sibling':
            if entry['receptacle_sibling'] == 1:
                label = 1
            else:
                label = 0
        else:
            if entry['receptacle_sibling'] == 1:
                label = 3
            else:
                label = entry['receptacle']

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
                'trajectory_id': trajectory_id, 'timestep': timestep, 'data_id': idx}

        # if there are transformations/augmentations apply them
        if self.transform is not None:
            self.transform(data)

        # create dict and return
        return data, label

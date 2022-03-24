import os

import numpy as np
import src.dataloaders.augmentations as A
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from src.dataloaders.roomr_dataset_utils import get_rearrange_task_spec
from src.shared.constants import CLASSES_TO_IGNORE, IMAGE_SIZE
from src.simulation.constants import ROOMR_CONTROLLER_COMMIT_ID
from src.simulation.environment import RearrangeTHOREnvironment
from src.simulation.module_box import GtBoxModule
from src.simulation.module_exploration import GtExplorationModule
from src.simulation.module_state_graph import StateGraphModule
from src.simulation.rearrange_utils import (load_rearrange_data_from_path,
                                            load_rearrange_meta_from_path)
from src.simulation.rearrangement_args import RearrangementArgs


class AgentDataGen(object):

    def __init__(
        self,
        rearrangement_args: RearrangementArgs
    ) -> None:
        super().__init__()
        if not os.path.exists(rearrangement_args.dump_dir):
            os.mkdir(rearrangement_args.dump_dir)
        self.dump_dir = rearrangement_args.dump_dir
        self.env = None
        self.roomr_metadata = load_rearrange_meta_from_path(
            rearrangement_args.data_split, rearrangement_args.roomr_meta_dir)
        self.reset(rearrangement_args=rearrangement_args)

    def reset(self, rearrangement_args=None):
        seed_everything(0)

        if rearrangement_args is not None:
            self.rearrangement_args = rearrangement_args

        # initialize modules based on flags
        self.box_module = None
        self.exploration_module = None

        # create env with basic controller
        if self.env is None:
            self.env = RearrangeTHOREnvironment(
                force_cache_reset=False,
                controller_kwargs={
                    'commit_id': ROOMR_CONTROLLER_COMMIT_ID,
                    'height': IMAGE_SIZE,
                    'width': IMAGE_SIZE,
                    'renderInstanceSegmentation': self.rearrangement_args.render_instance_segmentation,
                    'renderDepthImage': False,
                    'visibilityDistance': self.rearrangement_args.visibility_distance,
                    'quality': "Very Low"})

        # BOX MODULE
        if self.rearrangement_args.use_gt_boxes:
            box_frac_threshold = self.rearrangement_args.box_frac_threshold
            self.box_module = GtBoxModule(box_frac_threshold)
        else:
            raise NotImplementedError()

        # EXPLORATION MODULE
        if self.rearrangement_args.use_gt_exploration:
            split = self.rearrangement_args.data_split
            data = load_rearrange_data_from_path(
                split, self.rearrangement_args.roomr_dir)
            room_id = self.rearrangement_args.room_id
            dump_dir = self.rearrangement_args.dump_dir
            floor_plan = 'FloorPlan' + str(room_id)
            instance_id = self.rearrangement_args.instance_id
            exploration_strategy = self.rearrangement_args.gt_exploration_strategy
            num_steps = self.rearrangement_args.num_steps
            rotation_degrees = self.rearrangement_args.rotation_degrees
            task_spec = get_rearrange_task_spec(
                data, floor_plan, instance_id, split)
            metadata = self.roomr_metadata[f'{floor_plan}_{instance_id}']
            self.exploration_module = GtExplorationModule(
                self.env, task_spec, exploration_strategy, metadata, num_steps, rotation_degrees, room_id, instance_id, dump_dir)
        else:
            raise NotImplementedError()

        # BETWEEN TRAJECTORY CORRESPONDENCE MODULE
        if self.rearrangement_args.use_gt_object_matching:
            room_id = self.rearrangement_args.room_id
            dump_dir = self.rearrangement_args.dump_dir
            instance_id = self.rearrangement_args.instance_id
        else:
            raise NotImplementedError()

        assert self.env.controller == self.exploration_module.navi.controller

    def walkthrough_pipeline(self):
        self.explore_shared(True)


    def explore_shared(self, from_walkthrough):
        # initial state and initialize the representation
        thor_state = self.exploration_module.env.last_event
        if self.rearrangement_args.debug:
            self.exploration_module.dump_observation()

        thor_state = not None
        while True:
            thor_state, _ = self.exploration_module.take_action()

            if thor_state is None:
                break

            if self.rearrangement_args.debug:
                self.exploration_module.dump_observation()

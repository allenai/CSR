import itertools
import json
import os
from time import time

import numpy as np
import src.dataloaders.augmentations as A
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from src.dataloaders.roomr_dataset_utils import (find_waypoint_plan,
                                                 get_rearrange_task_spec)
from src.shared.constants import CLASSES_TO_IGNORE, IMAGE_SIZE
from src.simulation.constants import ROOMR_CONTROLLER_COMMIT_ID
from src.simulation.environment import RearrangeTHOREnvironment
from src.simulation.metrics import rand_metrics, rearrangement_metrics
from src.simulation.module_box import GtBoxModule, PredBoxModule
from src.simulation.module_exploration import GtExplorationModule, ReplayExplorationModule
from src.simulation.module_planner import PlannerModule
from src.simulation.module_relation_tracking import RelationTrackingModule
from src.simulation.module_state_graph import StateGraphModule
from src.simulation.rearrange_utils import (load_exploration_cache_dir, load_rearrange_data_from_path,
                                            load_rearrange_meta_from_path)
from src.simulation.rearrangement_args import RearrangementArgs
from src.simulation.shortest_path_navigator import ShortestPathNavigatorTHOR


class AgentRoomr(object):

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
        self.exploration_cache = None
        if not rearrangement_args.use_gt_exploration:
            self.exploration_cache = load_exploration_cache_dir(
                rearrangement_args.exploration_cache_dir)
        self.reset(rearrangement_args=rearrangement_args)

    def reset(self, rearrangement_args=None):
        seed_everything(0)

        if rearrangement_args is not None:
            self.rearrangement_args = rearrangement_args

        # initialize modules based on flags
        self.box_module = None
        self.exploration_module = None
        self.relation_tracking_module = None
        self.planner = None
        self.state_graph_module = None
        self.adjusted_rand_index_over_time = []

        self.use_gt_boxes = self.rearrangement_args.use_gt_boxes
        self.use_roi_feature_within_traj = self.rearrangement_args.use_roi_feature_within_traj
        self.use_roi_feature_between_traj = self.rearrangement_args.use_roi_feature_between_traj
        self.use_box_within_traj = self.rearrangement_args.use_box_within_traj

        moved_detection_counts = {}
        d = self.roomr_metadata[f'FloorPlan{rearrangement_args.room_id}_{rearrangement_args.instance_id}']['objects']
        for o in d:
            o_data = d[o]
            moved_detection_counts[o] = {
                'has_opened': o_data['has_opened'], 'count': 0}

        # create env with basic controller
        if self.env is None:
            self.env = RearrangeTHOREnvironment(
                force_cache_reset=False,
                controller_kwargs={
                    'commit_id': ROOMR_CONTROLLER_COMMIT_ID,
                    'height': IMAGE_SIZE,
                    'width': IMAGE_SIZE,
                    'renderInstanceSegmentation': self.rearrangement_args.render_instance_segmentation,
                    'renderDepthImage': self.rearrangement_args.debug,
                    'visibilityDistance': self.rearrangement_args.visibility_distance,
                    'quality': "Very Low"})

        # BOX MODULE
        self.box_module = None

        box_conf_threshold = self.rearrangement_args.box_conf_threshold
        box_frac_threshold = self.rearrangement_args.box_frac_threshold
        model_type = self.rearrangement_args.boxes_model_type
        model_path = self.rearrangement_args.boxes_model_path
        device_num = self.rearrangement_args.device_relation_tracking
        get_roi_features = self.rearrangement_args.use_roi_feature_within_traj or self.rearrangement_args.use_roi_feature_between_traj
        debug = self.rearrangement_args.debug

        BoxModule = None
        if self.rearrangement_args.use_gt_boxes:
            BoxModule = GtBoxModule
        else:
            BoxModule = PredBoxModule

        self.box_module = BoxModule(box_conf_threshold,
                                    box_frac_threshold,
                                    model_type,
                                    model_path,
                                    device_num,
                                    moved_detection_counts,
                                    get_roi_features,
                                    debug)

        # EXPLORATION MODULE
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

        if self.rearrangement_args.use_gt_exploration:
            metadata = self.roomr_metadata[f'{floor_plan}_{instance_id}']
            self.exploration_module = GtExplorationModule(
                self.env, task_spec, exploration_strategy, metadata, num_steps, rotation_degrees, room_id, instance_id, dump_dir)
        else:
            cache = self.exploration_cache[f'{room_id}_{instance_id}']
            self.exploration_module = ReplayExplorationModule(
                self.env, task_spec, cache, rotation_degrees, room_id, instance_id, dump_dir)

        # WITHIN TRAJECTORY CORRESPONDENCE MODULE
        use_roi_feature_within_traj = self.rearrangement_args.use_roi_feature_within_traj
        use_roi_feature_between_traj = self.rearrangement_args.use_roi_feature_between_traj
        self.relation_tracking_module = RelationTrackingModule(
            self.rearrangement_args.relation_tracking_model_path,
            self.rearrangement_args.object_tracking_model_path,
            self.rearrangement_args.averaging_strategy,
            self.rearrangement_args.device_relation_tracking,
            self.rearrangement_args.use_gt_relation_tracking,
            True,
            self.rearrangement_args.cos_sim_match_threshold,
            room_id,
            instance_id,
            dump_dir,
            use_roi_feature_within_traj,
            use_roi_feature_between_traj,
            self.rearrangement_args.debug)

        # BETWEEN TRAJECTORY CORRESPONDENCE MODULE
        room_id = self.rearrangement_args.room_id
        dump_dir = self.rearrangement_args.dump_dir
        instance_id = self.rearrangement_args.instance_id
        use_gt_object_matching = self.rearrangement_args.use_gt_object_matching

        self.planner = PlannerModule(
            self.env, room_id, instance_id, use_gt_object_matching, dump_dir)

        # AGENT POSE CORRESPONDENCE MODULE
        self.state_graph_module = StateGraphModule()

    def walkthrough_pipeline(self):
        self.explore_shared(True)

    def rearrange_room(self):
        self.exploration_module.reset(shuffle=True)
        self.relation_tracking_module.reset()
        self.state_graph_module.reset()
        self.box_module.reset()

    def unshuffle_pipeline(self):
        assert self.exploration_module.env.shuffle_called
        self.explore_shared(False)

        # at this point we need to compare the scene representations to figure out what moved and a plan
        self.planner.generate_plan(
            self.rearrangement_args.cos_sim_moved_threshold,
            self.rearrangement_args.cos_sim_object_threshold,
            self.rearrangement_args.debug)
        self.planner.execute_plan(self.rearrangement_args.debug)

    def get_metrics(self, with_error=False):
        metrics = rearrangement_metrics(
            self.env, self.planner, self.roomr_metadata, with_error)

        return metrics

    def explore_shared(self, from_walkthrough):
        # initial state and initialize the representation
        event = self.exploration_module.env.last_event
        seen_states = set()

        event_key = ShortestPathNavigatorTHOR.get_key(
            event.metadata['agent'])
        seen_states.add(event_key)

        if self.rearrangement_args.debug:
            self.exploration_module.dump_observation()

        grounded_state = self.relation_tracking_module.update_scene_representation(
            event,
            self.box_module,
        )

        _, ari = rand_metrics(self.relation_tracking_module.assignments,
                              self.relation_tracking_module.gt_assignments)
        self.adjusted_rand_index_over_time.append(ari)

        self.state_graph_module.add_edge(grounded_state, None)
        if self.rearrangement_args.debug:
            self.state_graph_module.dump_graph(from_walkthrough)
        event = not None
        while True:
            event, update_state_graph = self.exploration_module.take_action()

            if event is None:
                break

            last_action = event.metadata['lastAction']
            if self.rearrangement_args.debug:
                self.exploration_module.dump_observation()

            event_key = ShortestPathNavigatorTHOR.get_key(
                event.metadata['agent'])

            grounded_state = self.relation_tracking_module.update_scene_representation(
                event,
                self.box_module
            )

            seen_states.add(event_key)

            if update_state_graph:
                _, ari = rand_metrics(
                    self.relation_tracking_module.assignments, self.relation_tracking_module.gt_assignments)
                self.adjusted_rand_index_over_time.append(ari)
                # hack around fact that turns are split into 3 images
                self.state_graph_module.add_edge(
                    grounded_state, last_action)
                if self.rearrangement_args.debug:
                    self.state_graph_module.dump_graph(from_walkthrough)

        self.planner.store_representations(
            self.relation_tracking_module, self.state_graph_module, self.box_module, from_walkthrough)

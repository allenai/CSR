import json
import os
import random
from time import time
from typing import Dict

from PIL import Image
from ai2thor.controller import RECEPTACLE_OBJECTS
from src.dataloaders.roomr_dataset_utils import find_waypoint_plan
from src.models.exploration_model import StatefulExplorationModel
from src.shared.utils import check_none_or_empty, compute_3d_dist
from src.simulation.constants import EXPLORATION_ACTION_ORDER, OPENABLE_OBJECTS, PICKUPABLE_OBJECTS, REARRANGE_SIM_OBJECTS_COLOR_LOOKUP
from src.simulation.environment import (RearrangeTaskSpec,
                                        RearrangeTHOREnvironment)
from src.simulation.shortest_path_navigator import ShortestPathNavigatorTHOR
from src.simulation.utils import get_agent_map_data
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import \
    depth_frame_to_camera_space_xyz, camera_space_xyz_to_world_xyz
import torch
import trimesh


class GtExplorationModule(object):
    def __init__(self,
                 env: RearrangeTHOREnvironment,
                 task_spec: RearrangeTaskSpec,
                 exploration_strategy: str,
                 task_metadata: Dict,
                 num_steps: int,
                 rotation_degrees: int,
                 room_id: int,
                 instance_id: int,
                 dump_dir: str) -> None:
        super().__init__()
        self.env = env
        self.task_spec = task_spec
        self.exploration_strategy = exploration_strategy
        self.task_metadata = task_metadata
        self.num_steps = num_steps
        self.rotation_degrees = rotation_degrees
        self.room_id = room_id
        self.instance_id = instance_id
        self.dump_dir = dump_dir

        self.reset()

    def reset(self, shuffle=False):
        self.env.reset(self.task_spec,
                       force_axis_aligned_start=True)
        if shuffle:
            self.env.shuffle()
        self.navi = ShortestPathNavigatorTHOR(self.env.controller)
        self.navi_reachable_spots = self.navi.reachable_points_with_rotations_and_horizons()
        self.env.controller.step('Done')

        self.steps = 0
        self.turn_count = 0
        self.rollout = []
        self.last_action = None
        self.turn_direction = None
        self.targets = None
        self.target_key = None

        if self.exploration_strategy == 'fps':
            self.targets = [self.get_pose()['position']]
            self.target_key = self._get_fps_target_key()
        elif self.exploration_strategy == 'waypoint':
            self.targets = find_waypoint_plan(
                self.get_pose()['position'], self.task_metadata, self.env.shuffle_called)
            if len(self.targets):
                self.target_key = self.navi.get_key(self.targets.pop(0))
            else:
                self.target_key = None
        else:
            raise

    def take_action(self):
        if self.target_key is None:
            return None, False

        curr_key = self.navi.get_key(self.navi.last_event.metadata["agent"])
        self.last_action = None

        max_retries = 50
        retries = 0

        while 1:

            event = None
            update_state_graph = False

            assert retries < max_retries

            if self.turn_direction is not None:
                self.last_action = self.turn_direction
                event = self.navi.controller.step(
                    action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
                self.turn_count += self.rotation_degrees
                assert event.metadata['lastActionSuccess']
            else:
                self.last_action = self.navi.shortest_path_next_action(
                    curr_key, self.target_key)
                if self.last_action == 'RotateRight' or self.last_action == 'RotateLeft':
                    self.turn_direction = self.last_action
                    event = self.navi.controller.step(
                        action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
                    self.turn_count += self.rotation_degrees
                else:
                    event = self.navi.controller.step(
                        action=self.last_action, **self.env.physics_step_kwargs)
                    update_state_graph = True

                if not event.metadata['lastActionSuccess']:
                    self.navi.update_graph_with_failed_action(self.last_action)
                    retries += 1
                    continue

                self.steps += 1

            if self.turn_count == 90:
                self.turn_direction = None
                self.turn_count = 0
                update_state_graph = True

                # # turns are decomposed into 3 subactions but only want to increment steps once
                # self.steps += 1

            self.rollout.append(self.get_pose())

            if self.steps == self.num_steps:
                self.target_key = None
            elif self.at_goal():
                if self.exploration_strategy == 'fps':
                    self.targets.append(self.get_pose()['position'])
                    self.target_key = self.get_fps_target_key()
                elif self.exploration_strategy == 'waypoint':
                    if len(self.targets) != 0:
                        # print('hit a target!')
                        self.target_key = self.navi.get_key(
                            self.targets.pop(0))
                    else:
                        self.target_key = None
                        return event, True

            return event, update_state_graph

    def dump_observation(self):
        prefix = 'walkthrough'
        if self.env.shuffle_called:
            prefix = 'unshuffle'

        im = Image.fromarray(self.env.last_event.frame)
        im.save(
            f'{self.dump_dir}/{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.png', 'PNG')
        with open(os.path.join(self.dump_dir, f'{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.json'), 'w') as f:
            json.dump(self.env.last_event.metadata, f, indent=4)

    def at_goal(self):
        return self.navi.get_key(self.navi.last_event.metadata["agent"]) == self.target_key

    def get_pose(self):
        event = self.navi.controller.step(
            action='Done', **self.env.physics_step_kwargs
        )
        assert event.metadata["lastActionSuccess"]

        position = event.metadata["agent"]["position"]
        rotation = event.metadata["agent"]["rotation"]
        horizon = round(event.metadata["agent"]["cameraHorizon"], 2)

        return {'position': position, 'rotation': rotation, 'horizon': horizon}

    def _get_random_target_key(self):
        target_index = random.randint(0, len(self.navi_reachable_spots)-1)
        return self.navi.get_key(self.navi_reachable_spots[target_index])

    def _get_fps_target_key(self):
        avg = {'x': 0., 'y': 0., 'z': 0.}
        for t in self.targets:
            avg['x'] += t['x']
            avg['y'] += t['y']
            avg['z'] += t['z']

        avg['x'] /= len(self.targets)
        avg['y'] /= len(self.targets)
        avg['z'] /= len(self.targets)

        max_dist = 0.0
        max_arg = None

        for i, e in enumerate(self.navi_reachable_spots):
            d = compute_3d_dist(avg, e)
            if d > max_dist:
                max_arg = i
                max_dist = d

        return self.navi.get_key(self.navi_reachable_spots[max_arg])

class ReplayExplorationModule(object):
    def __init__(self,
                 env: RearrangeTHOREnvironment,
                 task_spec: RearrangeTaskSpec,
                 cache: Dict,
                 rotation_degrees: int,
                 room_id: int,
                 instance_id: int,
                 dump_dir: str) -> None:

        super().__init__()
        self.env = env
        self.task_spec = task_spec
        self.rotation_degrees = rotation_degrees
        self.room_id = room_id
        self.instance_id = instance_id
        self.dump_dir = dump_dir
        self.trajectories = cache
        self.trajectory = None

        self.reset()

    def reset(self, shuffle=False):
        self.env.reset(self.task_spec,
                       force_axis_aligned_start=True)

        self.trajectory = None
        if shuffle:
            self.env.shuffle()
            self.trajectory = self.trajectories['unshuffle'][:50]
        else:
            self.trajectory = self.trajectories['walkthrough'][:50]

        self.env.controller.step('Done')

        self.steps = 0
        self.turn_count = 0
        self.rollout = []
        self.last_action = None
        self.turn_direction = None

        # for result fig
        self.points = None
        self.colors = None

    def take_action(self):
        if self.steps >= len(self.trajectory) and self.turn_count == 0:
            return None, False

        self.last_action = None

        event = None
        update_state_graph = False

        if self.turn_direction is not None:
            self.last_action = self.turn_direction
            event = self.env.controller.step(
                action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
            self.turn_count += self.rotation_degrees
            if not event.metadata['lastActionSuccess']:
                raise ValueError(event.metadata['errorMessage'])
        else:
            # print(self.trajectory[self.steps])
            self.last_action = EXPLORATION_ACTION_ORDER[self.trajectory[self.steps]]

            if self.last_action == 'RotateRight' or self.last_action == 'RotateLeft':
                self.turn_direction = self.last_action
                event = self.env.controller.step(
                    action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
                self.turn_count += self.rotation_degrees
            else:
                event = self.env.controller.step(
                    action=self.last_action, **self.env.physics_step_kwargs)
                update_state_graph = True

            # we are replaying a trajectory so it should never fail
            if not event.metadata['lastActionSuccess']:
                raise ValueError(event.metadata['errorMessage'])

            self.steps += 1

        if self.turn_count == 90:
            self.turn_direction = None
            self.turn_count = 0
            update_state_graph = True

        self.rollout.append(self.get_pose())

        # if self.at_goal():
        #     return event, True

        return event, update_state_graph

    def dump_observation(self):
        prefix = 'walkthrough'
        if self.env.shuffle_called:
            prefix = 'unshuffle'

        im = Image.fromarray(self.env.last_event.frame)
        im.save(
            f'{self.dump_dir}/{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.png', 'PNG')
        with open(os.path.join(self.dump_dir, f'{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.json'), 'w') as f:
            json.dump(self.env.last_event.metadata, f, indent=4)

        camera_space_xyz = depth_frame_to_camera_space_xyz(
                depth_frame=torch.as_tensor(self.env.last_event.depth_frame), mask=None, fov=90
            )
        x = self.env.last_event.metadata['agent']['position']['x']
        y = self.env.last_event.metadata['agent']['position']['y']
        z = self.env.last_event.metadata['agent']['position']['z']

        world_points = camera_space_xyz_to_world_xyz(
            camera_space_xyzs=camera_space_xyz,
            camera_world_xyz=torch.as_tensor([x, y, z]),
            rotation=self.env.last_event.metadata['agent']['rotation']['y'],
            horizon=self.env.last_event.metadata['agent']['cameraHorizon'],
        ).reshape(3, self.env.controller.height, self.env.controller.width).permute(1, 2, 0)

        for mask_instance in self.env.last_event.instance_masks:
                detection_category = mask_instance.split('|')[0]
                if detection_category in REARRANGE_SIM_OBJECTS_COLOR_LOOKUP:
                # if (detection_category in PICKUPABLE_OBJECTS) or\
                #     (detection_category in OPENABLE_OBJECTS) or\
                #     (detection_category in RECEPTACLE_OBJECTS):
                # register this box in 3D
                    mask = self.env.last_event.instance_masks[mask_instance]
                    obj_points = world_points[mask]
                    obj_colors = torch.as_tensor(REARRANGE_SIM_OBJECTS_COLOR_LOOKUP[detection_category])
                    obj_colors.unsqueeze_(0)
                    obj_colors = obj_colors.repeat(obj_points.shape[0], 1)

                    if self.points is None:
                        self.points = obj_points
                        self.colors = obj_colors
                    else:
                        self.points = torch.cat((self.points, obj_points), 0)
                        self.colors = torch.cat((self.colors, obj_colors), 0)

                    assert self.points.shape == self.colors.shape

        rgba_colors = torch.ones(self.colors.shape[0], 4)
        rgba_colors[:, :3] = self.colors
        ply = trimesh.points.PointCloud(vertices=self.points.numpy(), colors=rgba_colors.numpy())
        ply.export(f'{self.dump_dir}/{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.ply')

        # top_down_data = get_agent_map_data(self.env.controller)
        # top_img = Image.fromarray(top_down_data['frame'])
        # top_img.save(os.path.join(
        #     self.dump_dir, f'{prefix}_{self.room_id}_{self.instance_id}_{self.steps}_top.png'), 'PNG')
        return

    def at_goal(self):
        tmp = self.steps + 1

        return tmp == len(self.trajectory)

    def get_pose(self):
        event = self.env.controller.step(
            action='Done', **self.env.physics_step_kwargs
        )
        assert event.metadata["lastActionSuccess"]

        position = event.metadata["agent"]["position"]
        rotation = event.metadata["agent"]["rotation"]
        horizon = round(event.metadata["agent"]["cameraHorizon"], 2)

        return {'position': position, 'rotation': rotation, 'horizon': horizon}

# class PredExplorationModule(object):
#     def __init__(self,
#                  env: RearrangeTHOREnvironment,
#                  task_spec: RearrangeTaskSpec,
#                  exploration_model_path: str,
#                  rotation_degrees: int,
#                  room_id: int,
#                  instance_id: int,
#                  dump_dir: str) -> None:

#         super().__init__()
#         self.env = env
#         self.task_spec = task_spec
#         self.rotation_degrees = rotation_degrees
#         self.room_id = room_id
#         self.instance_id = instance_id
#         self.dump_dir = dump_dir
#         if not check_none_or_empty(exploration_model_path):
#             self.relation_tracking_model = StatefulExplorationModel(exploration_model_path)
#         else:
#             raise ValueError('exploration_model_path should not be None or empty')

#         self.reset()

#     def reset(self, shuffle=False):
#         self.env.reset(self.task_spec,
#                        force_axis_aligned_start=True)

#         self.trajectory = None
#         self.relation_tracking_model.reset()
#         if shuffle:
#             self.env.shuffle()
#             self.trajectory = self.trajectories['unshuffle']
#         else:
#             self.trajectory = self.trajectories['walkthrough']

#         self.navi = ShortestPathNavigatorTHOR(self.env.controller)
#         self.navi_reachable_spots = self.navi.reachable_points_with_rotations_and_horizons()
#         self.env.controller.step('Done')

#         self.turn_count = 0
#         self.rollout = []
#         self.last_action = None
#         self.turn_direction = None

#     def take_action(self):

#         # curr_key = self.navi.get_key(self.navi.last_event.metadata["agent"])
#         self.last_action = None

#         event = None
#         update_state_graph = False

#         if self.turn_direction is not None:
#             self.last_action = self.turn_direction
#             event = self.navi.controller.step(
#                 action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
#             self.turn_count += self.rotation_degrees
#             assert event.metadata['lastActionSuccess']
#         else:
#             # print(self.trajectory[self.steps])
#             action = self.relation_tracking_model.get_action(self.env.controller)
#             if action is None:
#                 return None, False

#             self.last_action = EXPLORATION_ACTION_ORDER[action]

#             if self.last_action == 'RotateRight' or self.last_action == 'RotateLeft':
#                 self.turn_direction = self.last_action
#                 event = self.navi.controller.step(
#                     action=self.last_action, degrees=self.rotation_degrees, **self.env.physics_step_kwargs)
#                 self.turn_count += self.rotation_degrees
#             else:
#                 event = self.navi.controller.step(
#                     action=self.last_action, **self.env.physics_step_kwargs)
#                 update_state_graph = True

#             # we are replaying a trajectory so it should never fail
#             assert event.metadata['lastActionSuccess']

#         if self.turn_count == 90:
#             self.turn_direction = None
#             self.turn_count = 0
#             update_state_graph = True

#         self.rollout.append(self.get_pose())

#         # if self.at_goal():
#         #     return event, True

#         return event, update_state_graph

#     def dump_observation(self):
#         prefix = 'walkthrough'
#         if self.env.shuffle_called:
#             prefix = 'unshuffle'

#         im = Image.fromarray(self.env.last_event.frame)
#         im.save(
#             f'{self.dump_dir}/{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.png', 'PNG')
#         with open(os.path.join(self.dump_dir, f'{prefix}_{self.room_id}_{self.instance_id}_{self.steps}.json'), 'w') as f:
#             json.dump(self.env.last_event.metadata, f, indent=4)

#     def at_goal(self):
#         tmp = self.steps + 1

#         return tmp == len(self.trajectory)

#     def get_pose(self):
#         event = self.navi.controller.step(
#             action='Done', **self.env.physics_step_kwargs
#         )
#         assert event.metadata["lastActionSuccess"]

#         position = event.metadata["agent"]["position"]
#         rotation = event.metadata["agent"]["rotation"]
#         horizon = round(event.metadata["agent"]["cameraHorizon"], 2)

#         return {'position': position, 'rotation': rotation, 'horizon': horizon}

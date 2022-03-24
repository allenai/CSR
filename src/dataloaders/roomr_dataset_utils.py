import json
import os
import random
import shutil
from typing import Any, Dict, List, cast

import numpy as np
import src.dataloaders.augmentations as A
from src.shared.constants import IMAGE_SIZE
from src.shared.utils import compute_3d_dist
from src.simulation.constants import ROOMR_CONTROLLER_COMMIT_ID
from src.simulation.environment import (RearrangeTaskSpec,
                                        RearrangeTHOREnvironment)
from src.simulation.rearrange_utils import load_rearrange_data_from_path
from tqdm import tqdm


def get_waypoint(env, ip, reachable, name_to_meta):
    event = env.controller.step(
        action="GetInteractablePoses",
        objectId=ip["objectId"],
        positions=reachable,
        horizons=[-30, 0, 30],
        rotations=[0, 90, 180, 270],
        standings=[True]
    )

    obj_loc = name_to_meta[ip["name"]]["position"]

    possible_waypoints = event.metadata['actionReturn']
    if len(possible_waypoints) == 0:
        return None
    dists = [compute_3d_dist(obj_loc, w) for w in possible_waypoints]
    return possible_waypoints[np.argmin(dists)]


def get_rearrange_task_spec(roomr_data, floor_plan, index, stage):

    rearrangement_args = roomr_data[floor_plan][index]

    task_spec = RearrangeTaskSpec(scene=rearrangement_args['scene'],
                                  agent_position=rearrangement_args['agent_position'],
                                  agent_rotation=rearrangement_args['agent_rotation'],
                                  openable_data=rearrangement_args['openable_data'],
                                  starting_poses=rearrangement_args['starting_poses'],
                                  target_poses=rearrangement_args['target_poses'],
                                  stage=stage,
                                  runtime_sample=stage == 'train',)

    return task_spec


def find_meta(roomr_dirpath='/home/samirg/datasets/roomr/', stage='train', dump_dirpath='/home/samirg/datasets/roomr_meta2'):

    data = load_rearrange_data_from_path(
        stage, roomr_dirpath)
    if not os.path.exists(dump_dirpath):
        os.mkdir(dump_dirpath)

    meta_filepath = os.path.join(dump_dirpath, f'{stage}.json')

    env = RearrangeTHOREnvironment(force_cache_reset=stage != 'train', controller_kwargs={
        'commit_id': ROOMR_CONTROLLER_COMMIT_ID,
        'height': IMAGE_SIZE,
        'width': IMAGE_SIZE,
        'visibilityDistance': 1.5,
        'rotateStepDegrees': 90,
        'quality': "Very Low"})

    moved_dict = {}

    for scene_name in tqdm(data):
        for num, rearrangement_args in enumerate(data[scene_name]):
            assert num == rearrangement_args['index']
            room_instance = f'{scene_name}_{num}'
            moved_dict[room_instance] = {'objects': {}}

            task_spec = RearrangeTaskSpec(scene=rearrangement_args['scene'],
                                          agent_position=rearrangement_args['agent_position'],
                                          agent_rotation=rearrangement_args['agent_rotation'],
                                          openable_data=rearrangement_args['openable_data'],
                                          starting_poses=rearrangement_args['starting_poses'],
                                          target_poses=rearrangement_args['target_poses'],
                                          stage=stage,
                                          runtime_sample=stage == 'train',)

            # scene description that we are trying to recover
            env.reset(task_spec)
            walkthrough_reachable = env.controller.step(
                "GetReachablePositions").metadata["actionReturn"]
            walkthrough_name_to_meta = {
                e['name']: e for e in env.controller.last_event.metadata['objects']}
            walkthrough_id_to_name = {e['objectId']: e['name']
                                      for e in env.controller.last_event.metadata['objects']}

            env.shuffle()
            unshuffle_reachable = env.controller.step(
                "GetReachablePositions").metadata["actionReturn"]
            unshuffle_name_to_meta = {
                e['name']: e for e in env.controller.last_event.metadata['objects']}
            unshuffle_id_to_name = {e['objectId']: e['name']
                                    for e in env.controller.last_event.metadata['objects']}

            ips, gps, cps = env.poses
            pose_diffs = cast(
                List[Dict[str, Any]], env.compare_poses(
                    goal_pose=gps, cur_pose=cps)
            )

            # positions that are reachable before and after the shuffle
            reachable = [
                x for x in walkthrough_reachable if x in unshuffle_reachable]

            # gt what has moved
            pose_indices = []
            for i in range(len(pose_diffs)):
                shuffled_object_detected = False
                if pose_diffs[i]['iou'] is not None and pose_diffs[i]['iou'] < 0.5:
                    from_receptacle = None
                    to_receptacle = None

                    shuffled_object_detected = True

                    if walkthrough_name_to_meta[ips[i]["name"]]['parentReceptacles'] is not None:
                        from_receptacle = [walkthrough_id_to_name[e]
                                           for e in walkthrough_name_to_meta[ips[i]["name"]]['parentReceptacles']]
                    else:
                        print(
                            f'warning! no from receptacle for {ips[i]["name"]} in {room_instance}')

                    if unshuffle_name_to_meta[ips[i]["name"]]['parentReceptacles'] is not None:
                        to_receptacle = [unshuffle_id_to_name[e]
                                         for e in unshuffle_name_to_meta[ips[i]["name"]]['parentReceptacles']]
                    else:
                        print(
                            f'warning! no to receptacle for {ips[i]["name"]} in {room_instance}')

                    moved_dict[room_instance]['objects'][ips[i]["name"]] = {
                        "has_opened": False}
                    moved_dict[room_instance]['objects'][ips[i]
                                                         ["name"]]["from"] = from_receptacle
                    moved_dict[room_instance]['objects'][ips[i]
                                                         ["name"]]["to"] = to_receptacle
                    moved_dict[room_instance]['objects'][ips[i]["name"]
                                                         ]["position_dist"] = pose_diffs[i]["position_dist"]
                    moved_dict[room_instance]['objects'][ips[i]["name"]
                                                         ]["rotation_dist"] = pose_diffs[i]["rotation_dist"]

                if pose_diffs[i]['openness_diff'] is not None and pose_diffs[i]['openness_diff'] >= 0.2:
                    shuffled_object_detected = True

                    moved_dict[room_instance]['objects'][ips[i]["name"]] = {
                        "has_opened": True}
                    moved_dict[room_instance]['objects'][ips[i]["name"]
                                                         ]["openness_diff"] = pose_diffs[i]["openness_diff"]

                if shuffled_object_detected:
                    waypoint = get_waypoint(
                        env, ips[i], reachable, unshuffle_name_to_meta)
                    moved_dict[room_instance]['objects'][ips[i]
                                                         ["name"]]['unshuffle_waypoint'] = waypoint
                    pose_indices.append(i)

            moved_dict[room_instance]["position_diff_count"] = rearrangement_args['position_diff_count']
            moved_dict[room_instance]["open_diff_count"] = rearrangement_args['open_diff_count']

            # kinda a hack, but reset and then find the walkthrough waypoints
            env.reset(task_spec)
            for i in pose_indices:
                waypoint = get_waypoint(
                    env, gps[i], reachable, walkthrough_name_to_meta)
                moved_dict[room_instance]['objects'][gps[i]
                                                     ["name"]]['walkthrough_waypoint'] = waypoint

            # if stage != 'train':
            #     assert rearrangement_args['position_diff_count'] + rearrangement_args['open_diff_count'] == len(moved_dict[room_instance]['objects'])

    with open(meta_filepath, 'w') as f:
        json.dump(moved_dict, f, indent=4)


def find_waypoint_plan(start_location, instance_data, has_shuffled):

    def are_same(p1, p2, tol=0.001):
        sub_keys = ['x', 'y', 'z', 'rotation', 'horizon']
        for k in sub_keys:
            if abs(p1[k] - p2[k]) > tol:
                return False
        return p1['standing'] == p2['standing']
    # given a sequency of waypoints, compute a reasonable sequence to visit them
    all_points = []
    for k in instance_data['objects']:
        if instance_data['objects'][k]['walkthrough_waypoint'] is not None and instance_data['objects'][k]['unshuffle_waypoint'] is not None:
            walkthrough_waypoint = instance_data['objects'][k]['walkthrough_waypoint']
            unshuffle_waypoint = instance_data['objects'][k]['unshuffle_waypoint']

            if len(all_points):
                if not any([are_same(p, walkthrough_waypoint) for p in all_points]):
                    all_points.append(walkthrough_waypoint)
            else:
                all_points.append(walkthrough_waypoint)

            if len(all_points):
                if not any([are_same(p, unshuffle_waypoint) for p in all_points]):
                    all_points.append(unshuffle_waypoint)
            else:
                all_points.append(unshuffle_waypoint)

    sequence = []
    # greedy algo to determine a sequence of waypoints
    while len(all_points) != 0:
        dists = [compute_3d_dist(start_location, w) for w in all_points]
        sequence.append(all_points[np.argmin(dists)])
        all_points.remove(all_points[np.argmin(dists)])

    if has_shuffled:
        # random.shuffle(sequence)
        sequence = sequence[::-1]

    return sequence

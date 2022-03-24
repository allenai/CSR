import itertools
import json
import os
import random

import numpy as np
import src.dataloaders.augmentations as A
import torch
import torch.nn.functional as F
from ai2thor.controller import Controller
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from src.lightning.modules.moco2_module import MocoV2
from src.shared.constants import CLASSES_TO_IGNORE, IMAGE_SIZE
from src.shared.utils import (check_none_or_empty, load_lightning_inference,
                              render_adj_matrix)


class AgentObjNavExpert(object):

    def __init__(
        self,
        scene_name='FloorPlan1',
        image_size=IMAGE_SIZE,
        visibility_distance=1.5,
        random_start=True,
        trajectory=None,
        vision_model_path=None,
        rotation_step_degrees=90,
        dump_dir='./',
        box_frac_threshold=0.008,
        cos_sim_match_threshold=0.4,
        debug=False
    ) -> None:

        super().__init__()
        self.controller = Controller(
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=image_size,
            height=image_size,
            visibilityDistance=visibility_distance,
            rotateStepDegrees=rotation_step_degrees
        )

        self.model = None
        if not check_none_or_empty(vision_model_path):
            self.model = load_lightning_inference(
                vision_model_path, MocoV2).encoder_q

        self.reset(
            scene_name,
            image_size,
            visibility_distance,
            random_start,
            trajectory,
            rotation_step_degrees,
            dump_dir,
            box_frac_threshold,
            cos_sim_match_threshold,
            debug
        )

    def reset(
        self,
        scene_name,
        image_size,
        visibilityDistance,
        random_start,
        trajectory,
        rotation_step_degrees,
        dump_dir,
        box_frac_threshold,
        cos_sim_match_threshold,
        debug
    ):
        self.debug = debug
        self.thor_state = None
        self.box_frac_threshold = box_frac_threshold
        self.cos_sim_match_threshold = cos_sim_match_threshold

        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        self.dump_dir = dump_dir
        self.image_size = image_size

        self.rotation_step_degrees = rotation_step_degrees

        self.instance_map = {}
        self.gt_adjacency_matrix = np.zeros((0, 0))
        self.gt_assignments = []

        self.feature_bank = None
        self.feature_match_counts = None
        self.assignments = []
        self.cluster_meta = {}
        self.correct_assignments = 0
        self.total_assignments = 0

        if random_start and trajectory is not None:
            raise ValueError(
                'cannot set `random_start=True` and also pass a predefined `trajectory`')

        self.controller.reset(scene=scene_name,
                              width=image_size,
                              height=image_size,
                              visibilityDistance=visibilityDistance,
                              rotateStepDegrees=rotation_step_degrees)

        self.room_id = scene_name.split('FloorPlan')[1]

        event = self.controller.step(action="GetReachablePositions")
        assert event.metadata["lastActionSuccess"]

        self.reachable_spots = event.metadata["actionReturn"]
        self.steps = 0

        if random_start:
            i = random.randint(0, len(self.reachable_spots)-1)
            rot_y = random.choice(
                [i for i in range(0, 360, int(self.rotation_step_degrees))])

            event = self.controller.step(
                action='Teleport',
                position=self.reachable_spots[i],
                rotation=dict(x=0, y=rot_y, z=0),
                horizon=random.choice([-30, 0, 30, 60]),
                standing=True
            )
            assert event.metadata["lastActionSuccess"]

        self.rollout = []
        self.replay = False

        if trajectory is not None:
            event = self.controller.step(
                action='Teleport',
                position=trajectory[0]['position'],
                rotation=trajectory[0]['rotation'],
                horizon=trajectory[0]['horizon'],
                standing=True
            )

            assert event.metadata["lastActionSuccess"]

            self.rollout = trajectory
            self.replay = True

    def get_not_visible_object_ids(self):
        event = self.controller.step(
            action='Done',
        )
        assert event.metadata["lastActionSuccess"]

        objs = []

        for o in event.metadata['objects']:
            if not o['visible']:
                objs.append(o['objectId'])

        return objs

    def get_action(self, object_id):
        action = None

        event = self.controller.step(
            action='ObjectNavExpertAction',
            objectId=object_id
        )
        if not event.metadata["lastActionSuccess"]:
            raise ValueError('ObjectNavExpertAction failed')

        action = event.metadata["actionReturn"]

        return action

    def get_state(self):
        if self.thor_state is None:
            event = self.controller.step(
                action='Done',
            )
            assert event.metadata["lastActionSuccess"]
            self.thor_state = event

        return self.thor_state

    def take_action(self, object_id):

        action = None
        event = None

        if not self.replay:
            pose = self.get_pose()
            self.rollout.append(pose)
            action = self.get_action(object_id)
            if action is None:
                return action

            event = self.controller.step(
                action=action
            )

        else:
            if self.steps + 1 == len(self.rollout):
                return None
            action = "Teleport"
            event = self.controller.step(
                action="Teleport",
                position=self.rollout[self.steps+1]['position'],
                rotation=self.rollout[self.steps+1]['rotation'],
                horizon=int(self.rollout[self.steps+1]['horizon']),
                standing=True
            )

        if not event.metadata["lastActionSuccess"]:
            raise ValueError(f'{action} failed')

        self.thor_state = event
        self.steps += 1

        return action

    def move(self, object_ids):
        event = self.controller.step(
            action='Done',
        )
        assert event.metadata["lastActionSuccess"]
        include_set = set(object_ids)
        excluded_ids = []
        for obj in event.metadata['objects']:
            if obj['objectType'] not in include_set:
                excluded_ids.append(obj['objectId'])
            else:
                print(obj['objectId'])

        event = self.controller.step(action="InitialRandomSpawn",
                                     randomSeed=0,
                                     forceVisible=False,
                                     numPlacementAttempts=30,
                                     placeStationary=True,
                                     excludedObjectIds=excluded_ids
                                     )
        assert event.metadata["lastActionSuccess"]

    def get_pose(self):
        event = self.controller.step(
            action='Done',
        )
        assert event.metadata["lastActionSuccess"]

        position = event.metadata["agent"]["position"]
        rotation = event.metadata["agent"]["rotation"]
        horizon = round(event.metadata["agent"]["cameraHorizon"], 2)

        return {'position': position, 'rotation': rotation, 'horizon': horizon}

    def dump_observation(self):
        im = Image.fromarray(self.get_state().frame)
        im.save(f'{self.dump_dir}/{self.room_id}_1_{self.steps}.png', 'PNG')
        with open(os.path.join(self.dump_dir, f'{self.room_id}_1_{self.steps}.json'), 'w') as f:
            json.dump(self.thor_state.metadata, f, indent=4)

    def dump_clusters(self):
        assert self.feature_bank.shape[1] == len(self.cluster_meta)
        with open(f'{self.dump_dir}/cluster_meta_{self.room_id}_1_{self.steps}.json', 'w') as f:
            json.dump(self.cluster_meta, f, indent=4)
        torch.save(self.feature_bank, os.path.join(
            self.dump_dir, f'cluster_{self.room_id}_1_{self.steps}.pt'))

    def update_clusters(self):
        event = self.get_state()
        im = Image.fromarray(event.frame)

        step_instances = []
        new_count = 0
        boxes = {}

        for o in event.metadata['objects']:
            objecy_id = o['objectId']
            object_name = o['name']
            if objecy_id in event.instance_detections2D and o['visible']:
                if o['objectType'] in CLASSES_TO_IGNORE:
                    continue

                top = (event.instance_detections2D[objecy_id][0],
                       event.instance_detections2D[objecy_id][1])
                bottom = (event.instance_detections2D[objecy_id][2] - 1,
                          event.instance_detections2D[objecy_id][3] - 1)

                area = (bottom[0] - top[0]) * (bottom[1] - top[1])

                if area / (self.image_size * self.image_size) < self.box_frac_threshold:
                    continue

                if object_name not in self.instance_map:
                    self.instance_map[object_name] = len(self.instance_map)
                    self.gt_assignments.append([])
                    new_count += 1

                step_instances.append(object_name)
                self.gt_assignments[self.instance_map[object_name]].append(
                    object_name)

                box = Image.new('L', (self.image_size, self.image_size))
                tmp = ImageDraw.Draw(box)
                tmp.rectangle([top, bottom], fill="white")
                # if self.debug:
                box.save(
                    f'{self.dump_dir}/{self.room_id}_1_{self.steps}_{object_name}_box.png', 'PNG')
                boxes[object_name] = box

        if new_count > 0:
            # update gt adjacency matrix
            d_old = self.gt_adjacency_matrix.shape[0]
            d_new = d_old + new_count
            new_gt_adjacency_matrx = np.zeros((d_new, d_new))
            new_gt_adjacency_matrx[:d_old, :d_old] = self.gt_adjacency_matrix
            self.gt_adjacency_matrix = new_gt_adjacency_matrx

            # fill in the gt adjacency matrix
            step_pairs = list(itertools.product(step_instances, repeat=2))
            for p in step_pairs:
                i = self.instance_map[p[0]]
                j = self.instance_map[p[1]]
                self.gt_adjacency_matrix[i, j] = 1
                self.gt_adjacency_matrix[j, i] = 1

        if len(step_instances) == 0:
            # case where there are no detections, just want to return the action
            return

        # run inference on the self-features
        query_features = []
        for s in step_instances:

            data = {'mask_1': boxes[s].copy(),
                    'mask_2': boxes[s].copy(),
                    'image': im.copy()}

            # if there are transformations/augmentations apply them
            A.TestTransform(data)

            x = torch.cat((data['image'], data['mask_1'],
                           data['mask_2']), 0).unsqueeze(0)

            feat = self.model(x)
            query_features.append(F.normalize(feat, dim=1))
        query_features = torch.cat(tuple(query_features), 0)

        # start with all features being unmatched with the history
        unmatched_queries = set(
            [i for i in range(query_features.shape[0])])

        if self.feature_bank is None:
            # if there is no feature bank, then the features we create the bank
            self.feature_bank = torch.transpose(query_features, 0, 1)

            # keep track of the number of matches per feature in the bank for weighted averages
            self.feature_match_counts = torch.ones(
                self.feature_bank.shape[1])

            # initialize the pred assignments
            self.assignments = [[s] for s in step_instances]

            # create data structure to keep track of cluster to instance name matching (for metrics)
            self.cluster_meta = {i: {s: 1, 'representative': s}
                                 for i, s in enumerate(step_instances)}

            # for first step all asignments are correct assignments (assuiming GT boxes)
            self.total_assignments += len(step_instances)
            self.correct_assignments += len(step_instances)
        else:
            # create a reward matrix between current observation and the feature bank
            sim = torch.matmul(query_features, self.feature_bank)

            # hungarian matching to get the maximal assignment
            query_idx, history_idx = linear_sum_assignment(
                sim.numpy(), maximize=True)

            assert len(query_idx) == len(history_idx)

            # add the number of queries (denom for a metric)
            self.total_assignments += query_features.shape[0]

            # get the identies of the clusters before updating
            prev_representatives = set(
                [self.cluster_meta[i]['representative'] for i in self.cluster_meta])

            for i in range(len(query_idx)):
                cluster_number = history_idx[i]
                if sim[query_idx[i], history_idx[i]] > self.cos_sim_match_threshold:
                    # considered a match if the sim is greater than the threshold

                    # remove from the unmatched queries set
                    unmatched_queries.remove(query_idx[i])

                    # weighted average to integrate the query feature into the history
                    self.feature_bank[:, cluster_number] = self.feature_bank[:, cluster_number] * \
                        self.feature_match_counts[cluster_number] + \
                        query_features[query_idx[i]]
                    self.feature_match_counts[cluster_number] += 1
                    self.feature_bank[:,
                                      cluster_number] /= self.feature_match_counts[cluster_number]

                    # renormalize
                    self.feature_bank[:, cluster_number] = F.normalize(
                        self.feature_bank[:, cluster_number], dim=0)

                    # add the gt label of the assignment to this cluster for metrics
                    assigned_label = step_instances[query_idx[i]]
                    self.assignments[cluster_number].append(assigned_label)

                    # find the current representative of the cluster
                    representative_label = self.cluster_meta[cluster_number]['representative']

                    if assigned_label in self.cluster_meta[cluster_number]:
                        self.cluster_meta[cluster_number][assigned_label] += 1
                        if assigned_label == representative_label:
                            # we are assining the feature the a cluster with the same gt label, this is good
                            self.correct_assignments += 1
                    else:
                        # here we are adding to a cluster that has never before seen this instance, not good
                        self.cluster_meta[cluster_number][assigned_label] = 1

                    if self.cluster_meta[cluster_number][representative_label] <= self.cluster_meta[cluster_number][assigned_label]:
                        # update the gt label identity of the cluster for purposes of metrics
                        # NOTE: this is fine to do in the loop as the linear assignment ensures each cluster_number is unique for the update
                        self.cluster_meta[cluster_number]['representative'] = assigned_label

            # get the queries that have not matched
            unmatched_queries = list(unmatched_queries)

            for u in unmatched_queries:
                if step_instances[u] not in prev_representatives:
                    # case where we correctly assign a new cluster for this instance
                    self.correct_assignments += 1

            for u in unmatched_queries:
                # add a cluster for each new unmatched query
                self.assignments.append([step_instances[u]])
                self.cluster_meta[len(self.cluster_meta)] = {
                    step_instances[u]: 1, 'representative': step_instances[u]}

            # append features to the feature bank
            new_features = torch.transpose(
                query_features[unmatched_queries], 0, 1)
            self.feature_bank = torch.cat(
                (self.feature_bank, new_features), 1)
            self.feature_match_counts = torch.cat((self.feature_match_counts, torch.ones(
                len(unmatched_queries))), 0)

    def rand_metrics(self):
        gt_labels = []
        for i, c in enumerate(self.gt_assignments):
            gt_labels += [i] * len(c)

        pred_labels = []
        for i, c in enumerate(self.assignments):
            pred_labels += [i] * len(c)

        return rand_score(gt_labels, pred_labels), adjusted_rand_score(gt_labels, pred_labels)

    def atomic_mistake_metric(self):
        if self.total_assignments == 0:
            return 1.0
        return self.correct_assignments / float(self.total_assignments)

    def num_objects_mape_metric(self):
        if self.total_assignments == 0:
            return 0.0
        return abs(len(self.gt_assignments) - len(self.assignments)) / float(len(self.gt_assignments))

    def dump_gt_adjacency_matrix(self):
        row_labels = [k for k, _ in sorted(
            self.instance_map.items(), key=lambda item: item[1])]
        mat = render_adj_matrix(self.gt_adjacency_matrix, row_labels)
        sim_mat = Image.fromarray(mat, 'RGB')
        sim_mat.save(
            f'{self.dump_dir}/{self.room_id}_1_{self.steps-1}_adj.png')
        sim_mat.close()

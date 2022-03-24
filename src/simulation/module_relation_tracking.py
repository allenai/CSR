import itertools
import json
import time

import numpy as np
import src.dataloaders.augmentations as A
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from src.lightning.modules import moco2_module_old
from src.lightning.modules import moco2_module
from src.shared.constants import IMAGE_SIZE
from src.shared.utils import (check_none_or_empty, get_device,
                              load_lightning_inference, render_adj_matrix)
from src.simulation.module_box import GtBoxModule
from src.simulation.state import State
from src.simulation.utils import get_openable_objects, get_pickupable_objects
from torchvision.transforms.transforms import ToTensor


class RelationTrackingModule(object):
    def __init__(
            self,
            relation_tracking_model_path,
            object_tracking_model_path,
            averaging_strategy,
            device_relation_tracking,
            use_gt_matches,
            use_gt_are_close,
            cos_sim_match_threshold,
            room_id,
            instance_id,
            dump_dir,
            use_roi_feature_within_traj,
            use_roi_feature_between_traj,
            debug) -> None:

        super().__init__()

        self.relation_tracking_model = None
        self.object_tracking_model = None
        self.device = get_device(device_relation_tracking)
        self.debug = debug
        self.room_id = room_id
        self.instance_id = instance_id
        self.dump_dir = dump_dir
        self.use_roi_feature_within_traj = use_roi_feature_within_traj
        self.use_roi_feature_between_traj = use_roi_feature_between_traj

        if not check_none_or_empty(relation_tracking_model_path):
            self.relation_tracking_model = load_lightning_inference(
                relation_tracking_model_path, moco2_module_old.MocoV2).encoder_q.to(self.device)
        else:
            raise ValueError(
                'relation_tracking_model_path should never be None or empty')

        if not check_none_or_empty(object_tracking_model_path):
            self.object_tracking_model = load_lightning_inference(
                object_tracking_model_path, moco2_module.MocoV2).encoder_q.to(self.device)
        else:
            raise ValueError(
                'object_tracking_model_path should never be None or empty')

        self.averaging_strategy = averaging_strategy
        self.use_gt_matches = use_gt_matches
        self.use_gt_are_close = use_gt_are_close

        if self.use_gt_matches:
            self.cos_sim_match_threshold = None
        else:
            self.cos_sim_match_threshold = cos_sim_match_threshold
        self.reset()

    def reset(self):
        self.update_count = 0
        self.instance_map = {}
        self.gt_adjacency_matrix = np.zeros((0, 0))
        self.gt_assignments = []

        self.object_bank = None
        self.object_match_counts = None

        self.feature_bank = None
        self.feature_match_counts = None
        self.relationship_bank = None
        self.relationship_match_counts = {}
        self.assignments = []
        self.cluster_meta = {}
        self.state_graph = None
        self.correct_assignments = 0
        self.total_assignments = 0

        self.box_timer = []
        self.csr_timer = []
        self.obj_timer = []
        self.matching_timer = []

    def update_scene_representation(
        self,
        event,
        box_module,
    ):
        im = Image.fromarray(event.frame)



        tic = time.perf_counter()
        step_instances, boxes, interaction_points, areas, roi_features = box_module.get_boxes(event)
        toc = time.perf_counter()
        self.box_timer.append(toc-tic)

        # cluster book keeping
        new_count = 0
        for name in step_instances:
            if name not in self.instance_map:
                self.instance_map[name] = len(self.instance_map)
                self.gt_assignments.append([])
                new_count += 1

            self.gt_assignments[self.instance_map[name]].append(name)

        pickupable_objects = set(get_pickupable_objects(event))
        openable_objects = set(get_openable_objects(event))
        agent_position = event.metadata['agent']['position']
        agent_rotation = event.metadata['agent']['rotation']
        agent_horizon = event.metadata['agent']['cameraHorizon']

        if new_count > 0:
            # update gt adjacency matrix
            dim_old = self.gt_adjacency_matrix.shape[0]
            dim_new = dim_old + new_count
            new_gt_adjacency_matrx = np.zeros((dim_new, dim_new))
            new_gt_adjacency_matrx[:dim_old,
                                   :dim_old] = self.gt_adjacency_matrix
            self.gt_adjacency_matrix = new_gt_adjacency_matrx

            # fill in the gt adjacency matrix
            step_pairs = list(itertools.product(step_instances, repeat=2))
            for p in step_pairs:
                i = self.instance_map[p[0]]
                j = self.instance_map[p[1]]
                self.gt_adjacency_matrix[i, j] = 1
                self.gt_adjacency_matrix[j, i] = 1

        if len(step_instances) == 0:
            # case where there are no detections, just want to return
            # Have to save in the state graph
            return State([], {}, {}, [], [], [], {}, im, agent_position, agent_rotation, agent_horizon)

        # run inference on the self-features
        query_features = []
        step_instace_to_index = {}
        for step_index, step_instance in enumerate(step_instances):

            step_instace_to_index[step_instance] = step_index

        edge_features = {}

        edge_pairings = list(itertools.permutations(boxes.keys(), 2))
        num_self = len(step_instances)
        self_pairings = [(i, i) for i in boxes]
        keys = self_pairings + edge_pairings

        x = self.create_batch(keys, boxes, im)
        A.TestTransform(x)
        x_instance = torch.cat((x['image'], x['mask_1'], x['mask_2']),
                            1).to(self.device)
        x_object = torch.cat((x['image'][:num_self], x['mask_1'][:num_self], x['mask_2'][:num_self]),
                            1).to(self.device)

        if self.use_roi_feature_within_traj:
            query_features = roi_features.cpu()
        else:
            feat_instance = None
            i = 0
            tic = time.perf_counter()
            while i < x_instance.shape[0]:
                if feat_instance is None:
                    feat_instance = self.relation_tracking_model(x_instance[i:i+100])
                else:
                    feat_instance = torch.cat((feat_instance, self.relation_tracking_model(x_instance[i:i+100])), 0)
                i += 100
            toc = time.perf_counter()
            self.csr_timer.append(toc-tic)

            feat_instance = F.normalize(feat_instance, dim=1).cpu()
            query_features = feat_instance[:num_self]

            for i, pairing in enumerate(edge_pairings):
                edge_features[pairing] = feat_instance[i + num_self]


        object_features = None
        if self.use_roi_feature_between_traj:
            object_features = roi_features.cpu()
        else:
            tic = time.perf_counter()
            feat_object = self.object_tracking_model(x_object)
            toc = time.perf_counter()
            self.obj_timer.append(toc-tic)

            object_features = F.normalize(feat_object, dim=1).cpu()

        assert object_features.shape[0] == query_features.shape[0]

        state = None
        tic = time.perf_counter()

        if self.feature_bank is None:
            state = self.initialize_scene_representation(
                query_features,
                edge_features,
                object_features,
                step_instances,
                im,
                agent_position,
                agent_rotation,
                agent_horizon,
                boxes,
                pickupable_objects,
                openable_objects,
                interaction_points,
                areas)
        else:
            if self.use_gt_matches:
                state = self.match_scene_representation_gt(
                    query_features,
                    edge_features,
                    object_features,
                    step_instances,
                    im,
                    agent_position,
                    agent_rotation,
                    agent_horizon,
                    boxes,
                    pickupable_objects,
                    openable_objects,
                    interaction_points,
                    areas)
            else:
                state = self.match_scene_representation_pred(
                    query_features,
                    edge_features,
                    object_features,
                    step_instances,
                    im,
                    agent_position,
                    agent_rotation,
                    agent_horizon,
                    boxes,
                    pickupable_objects,
                    openable_objects,
                    interaction_points,
                    areas)

        toc = time.perf_counter()
        self.matching_timer.append(toc-tic)

        assert self.relationship_bank.shape[0] == self.relationship_bank.shape[1]
        assert self.relationship_bank.shape[2] == self.feature_bank.shape[0]
        assert self.relationship_bank.shape[0] == self.feature_bank.shape[1]

        # update the relationship with the main diagonal self features
        for i in range(self.feature_bank.shape[1]):
            self.relationship_bank[i, i] = self.feature_bank[:, i]

        return state

    def create_batch(self, keys, boxes, im):
        mask_1 = torch.zeros((len(keys), 1, IMAGE_SIZE, IMAGE_SIZE))
        mask_2 = torch.zeros((len(keys), 1, IMAGE_SIZE, IMAGE_SIZE))
        image = torch.zeros((len(keys), 3, IMAGE_SIZE, IMAGE_SIZE))
        t = ToTensor()
        tensor_image = t(im)
        for i, k in enumerate(keys):
            mask_1[i] = boxes[k[0]]
            mask_2[i] = boxes[k[1]]
            image[i] = torch.clone(tensor_image)

        return {'mask_1': mask_1, 'mask_2': mask_2, 'image': image}

    def initialize_scene_representation(
            self,
            query_features,
            edge_features,
            object_features,
            step_instances,
            im,
            agent_position,
            agent_rotation,
            agent_horizon,
            boxes,
            pickupable_objects,
            openable_objects,
            interaction_points,
            areas):
        if self.debug:
            self.dump_features_and_labels(query_features, edge_features, step_instances)
        self.update_count += 1

        # if there is no feature bank, then the features we create the bank
        self.feature_bank = torch.transpose(query_features, 0, 1)
        self.object_bank = torch.transpose(object_features, 0, 1)

        # also initialize a separate data structure for the edges
        self.relationship_bank = torch.zeros(
            query_features.shape[0], query_features.shape[0], query_features.shape[1])
        for pair in edge_features:
            self.relationship_bank[pair[0], pair[1]] = edge_features[pair]

        # keep track of the number of matches per feature in the bank for weighted averages
        self.feature_match_counts = torch.ones(
            self.feature_bank.shape[1])
        self.object_match_counts = torch.ones(
            self.object_bank.shape[1])

        # initialize the pred assignments
        self.assignments = [[s] for s in step_instances]

        # create data structure to keep track of cluster to instance name matching (for metrics)
        self.cluster_meta = {i: {s: 1, 'representative': s}
                             for i, s in enumerate(step_instances)}

        cluster_idx_to_name = {i: s for i, s in enumerate(step_instances)}

        # for first step all assignments are correct assignments (assuming GT boxes)
        self.total_assignments += len(step_instances)
        self.correct_assignments += len(step_instances)

        cluster_idxs = [i for i in self.cluster_meta]

        pickupable_bools = []
        openable_bools = []
        pickupable_points = {}
        openable_points = {}

        for i in self.cluster_meta:

            if cluster_idx_to_name[i] in pickupable_objects:
                pickupable_bools.append(i)
                pickupable_points[i] = interaction_points[i]

            if cluster_idx_to_name[i] in openable_objects:
                openable_bools.append(i)
                openable_points[i] = interaction_points[i]

        # add state to graph
        state = State(cluster_idxs, pickupable_points, openable_points, pickupable_bools,
                      openable_bools, boxes, areas, im, agent_position, agent_rotation, agent_horizon)

        return state

    def match_scene_representation_gt(
            self,
            query_features,
            edge_features,
            object_features,
            step_instances,
            im,
            agent_position,
            agent_rotation,
            agent_horizon,
            boxes,
            pickupable_objects,
            openable_objects,
            interaction_points,
            areas) -> State:

        if self.debug:
            self.dump_features_and_labels(query_features, edge_features, step_instances)
        self.update_count += 1

        # add the number of queries (denom for a metric)
        self.total_assignments += query_features.shape[0]

        name_to_cluster_idx = {
            self.cluster_meta[i]['representative']: i for i in self.cluster_meta}

        det_idx_to_cluster_idx = {
            i: None for i in range(query_features.shape[0])}

        num_new_clusters = 0

        for det_idx, name in enumerate(step_instances):
            if name in name_to_cluster_idx:
                cluster_idx = name_to_cluster_idx[name]
                det_idx_to_cluster_idx[det_idx] = cluster_idx

                self.cluster_meta[cluster_idx][name] += 1

                if self.averaging_strategy == 'weighted':
                    # weighted average to integrate the query feature into the history
                    self.weighted_average_self_feature(
                        cluster_idx, query_features, object_features, det_idx)
                else:
                    # unweighted average, which has the affect of weighting newer observations more
                    self.unweighted_average_self_feature(
                        cluster_idx, query_features, object_features, det_idx)

                # renormalize
                self.feature_bank[:, cluster_idx] = F.normalize(
                    self.feature_bank[:, cluster_idx], dim=0)
                self.object_bank[:, cluster_idx] = F.normalize(
                    self.object_bank[:, cluster_idx], dim=0)

                # add the gt label of the assignment to this cluster for metrics
                self.assignments[cluster_idx].append(name)

            else:
                # add a cluster for each new unmatched query
                num_new_clusters += 1
                self.assignments.append([name])
                det_idx_to_cluster_idx[det_idx] = len(
                    self.cluster_meta)

                self.cluster_meta[len(self.cluster_meta)] = {
                    name: 1, 'representative': name}

                # append features to the feature bank
                new_features = query_features[det_idx].unsqueeze(-1)
                self.feature_bank = torch.cat(
                    (self.feature_bank, new_features), 1)
                self.feature_match_counts = torch.cat(
                    (self.feature_match_counts, torch.ones(1)), 0)

                new_features = object_features[det_idx].unsqueeze(-1)
                self.object_bank = torch.cat(
                    (self.object_bank, new_features), 1)
                self.object_match_counts = torch.cat(
                    (self.object_match_counts, torch.ones(1)), 0)

        cluster_idx_to_name = {
            i: self.cluster_meta[i]['representative'] for i in self.cluster_meta}

        # expand the relationship bank as necessary
        if num_new_clusters != 0:
            n_old = self.relationship_bank.shape[0]
            n_new = n_old + num_new_clusters
            tmp = torch.zeros(n_new, n_new, query_features.shape[1])
            tmp[:n_old, :n_old, :] = self.relationship_bank
            self.relationship_bank = tmp

        # create the state representation with references to the scene object representation
        cluster_idxs = list(det_idx_to_cluster_idx.values())
        pickupable_bools = []
        openable_bools = []
        pickupable_points = {}
        openable_points = {}

        for det_idx in det_idx_to_cluster_idx:
            cluster_idx = det_idx_to_cluster_idx[det_idx]
            if cluster_idx_to_name[cluster_idx] in pickupable_objects:
                pickupable_bools.append(cluster_idx)
                pickupable_points[cluster_idx] = interaction_points[det_idx]

            if cluster_idx_to_name[cluster_idx] in openable_objects:
                openable_bools.append(cluster_idx)
                openable_points[cluster_idx] = interaction_points[det_idx]

        state = State(cluster_idxs, pickupable_points, openable_points, pickupable_bools,
                      openable_bools, boxes, areas, im, agent_position, agent_rotation, agent_horizon)
        # print(f'pickupable: {pickupable_bools}')
        # # print(f'openable: {openable_bools}')
        # print(cluster_idx_to_name)
        # print('-' * 30)

        for pair in edge_features:
            # fill in the edge feature representations
            ith, jth = det_idx_to_cluster_idx[pair[0]
                                              ], det_idx_to_cluster_idx[pair[1]]

            # NOTE: could be the case that two detections might coor
            # to the same cluster if pred boxes being used
            if ith == jth:
                continue

            if (ith, jth) not in self.relationship_match_counts:
                # norm should be 1, so if this is the case we have a new relation and need to just fill it with the edge feature
                self.relationship_bank[ith, jth] = edge_features[pair]
                self.relationship_match_counts[(ith, jth)] = 1
            elif self.averaging_strategy == 'weighted':
                raise NotImplementedError('gotta write this still')
            else:
                self.relationship_match_counts[(ith, jth)] += 1
                self.relationship_bank[ith, jth] = (
                    self.relationship_bank[ith, jth] + edge_features[pair]) / 2
                self.relationship_bank[ith, jth] = F.normalize(
                    self.relationship_bank[ith, jth], dim=0)

        self.correct_assignments += len(step_instances)

        return state

    def match_scene_representation_pred(
            self,
            query_features,
            edge_features,
            object_features,
            step_instances,
            im,
            agent_position,
            agent_rotation,
            agent_horizon,
            boxes,
            pickupable_objects,
            openable_objects,
            interaction_points,
            areas):

        if self.debug:
            self.dump_features_and_labels(query_features, edge_features, step_instances)
        self.update_count += 1

        # start with all features being unmatched with the history
        unmatched_queries = set([i for i in range(query_features.shape[0])])

        # create a reward matrix between current observation and the feature bank
        sim = torch.matmul(query_features, self.feature_bank)

        # hungarian matching to get the maximal assignment
        query_idx, history_idx = linear_sum_assignment(
            sim.numpy(), maximize=True)

        assert len(query_idx) == len(history_idx)

        # add the number of queries (denom for a metric)
        self.total_assignments += query_features.shape[0]

        # get the identities of the clusters before updating (for metrics)
        prev_representatives = set(
            [self.cluster_meta[i]['representative'] for i in self.cluster_meta])

        det_idx_to_cluster_idx = {
            i: None for i in range(query_features.shape[0])}

        for i in range(len(query_idx)):
            cluster_number = history_idx[i]
            if sim[query_idx[i], history_idx[i]] > self.cos_sim_match_threshold:
                # considered a match if the sim is greater than the threshold
                det_idx_to_cluster_idx[query_idx[i]] = cluster_number

                # remove from the unmatched queries set
                unmatched_queries.remove(query_idx[i])

                if self.averaging_strategy == 'weighted':
                    # weighted average to integrate the query feature into the history
                    self.weighted_average_self_feature(
                        cluster_number, query_features, object_features, query_idx[i])
                else:
                    # unweighted average, which has the affect of weighting newer observations more
                    self.unweighted_average_self_feature(
                        cluster_number, query_features, object_features, query_idx[i])

                # re-normalize
                self.feature_bank[:, cluster_number] = F.normalize(
                    self.feature_bank[:, cluster_number], dim=0)
                self.object_bank[:, cluster_number] = F.normalize(
                    self.object_bank[:, cluster_number], dim=0)

                # add the gt label of the assignment to this cluster for metrics
                assigned_label = step_instances[query_idx[i]]
                self.assignments[cluster_number].append(assigned_label)

                # find the current representative of the cluster
                representative_label = self.cluster_meta[cluster_number]['representative']

                if assigned_label in self.cluster_meta[cluster_number]:
                    self.cluster_meta[cluster_number][assigned_label] += 1
                    if assigned_label == representative_label:
                        # we are assigning the feature the a cluster with the same gt label, this is good
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
            det_idx_to_cluster_idx[u] = len(self.cluster_meta)
            self.cluster_meta[len(self.cluster_meta)] = {
                step_instances[u]: 1, 'representative': step_instances[u]}

        # expand the relationship bank as necessary
        num_new_clusters = len(unmatched_queries)
        if num_new_clusters != 0:
            n_old = self.relationship_bank.shape[0]
            n_new = n_old + num_new_clusters
            tmp = torch.zeros(n_new, n_new, query_features.shape[1])
            tmp[:n_old, :n_old, :] = self.relationship_bank
            self.relationship_bank = tmp

        # append features to the feature bank
        new_features = torch.transpose(
            query_features[unmatched_queries], 0, 1)
        self.feature_bank = torch.cat(
            (self.feature_bank, new_features), 1)
        self.feature_match_counts = torch.cat((self.feature_match_counts, torch.ones(
            len(unmatched_queries))), 0)

        new_features = torch.transpose(
            object_features[unmatched_queries], 0, 1)
        self.object_bank = torch.cat(
            (self.object_bank, new_features), 1)
        self.object_match_counts = torch.cat((self.object_match_counts, torch.ones(
            len(unmatched_queries))), 0)

        for pair in edge_features:
            # fill in the edge feature representations
            ith, jth = det_idx_to_cluster_idx[pair[0]
                                              ], det_idx_to_cluster_idx[pair[1]]
            assert ith != jth
            if (ith, jth) not in self.relationship_match_counts:
                # norm should be 1, so if this is the case we have a new relation and need to just fill it with the edge feature
                self.relationship_bank[ith, jth] = edge_features[pair]
                self.relationship_match_counts[(ith, jth)] = 1
            elif self.averaging_strategy == 'weighted':
                raise NotImplementedError('gotta write this still')
            else:
                self.relationship_match_counts[(ith, jth)] += 1
                self.relationship_bank[ith, jth] = (
                    self.relationship_bank[ith, jth] + edge_features[pair]) / 2
                self.relationship_bank[ith, jth] = F.normalize(
                    self.relationship_bank[ith, jth], dim=0)

        cluster_idxs = list(det_idx_to_cluster_idx.values())
        pickupable_bools = []
        openable_bools = []
        pickupable_points = {}
        openable_points = {}

        cluster_idx_to_name = {
            i: self.cluster_meta[i]['representative'] for i in self.cluster_meta}

        for det_idx in det_idx_to_cluster_idx:
            cluster_idx = det_idx_to_cluster_idx[det_idx]

            if cluster_idx_to_name[cluster_idx] in pickupable_objects and det_idx in interaction_points:
                pickupable_bools.append(cluster_idx)
                pickupable_points[cluster_idx] = interaction_points[det_idx]

            if cluster_idx_to_name[cluster_idx] in openable_objects and det_idx in interaction_points:
                openable_bools.append(cluster_idx)
                openable_points[cluster_idx] = interaction_points[det_idx]

        # print(f'pickupable: {pickupable_bools}')
        # # print(f'openable: {openable_bools}')
        # print(cluster_idx_to_name)
        # print('-' * 30)

        state = State(cluster_idxs, pickupable_points, openable_points, pickupable_bools,
                      openable_bools, boxes, areas, im, agent_position, agent_rotation, agent_horizon)

        return state

    def weighted_average_self_feature(self, cluster_number, query_features, object_featrues, instance_number):
        # weighted average to integrate the query feature into the history
        self.feature_bank[:, cluster_number] = self.feature_bank[:, cluster_number] * \
            self.feature_match_counts[cluster_number] + \
            query_features[instance_number]
        self.feature_match_counts[cluster_number] += 1
        self.feature_bank[:,
                          cluster_number] /= self.feature_match_counts[cluster_number]

        self.object_bank[:, cluster_number] = self.object_bank[:, cluster_number] * \
            self.object_match_counts[cluster_number] + \
            object_featrues[instance_number]
        self.object_match_counts[cluster_number] += 1
        self.object_bank[:,
                         cluster_number] /= self.object_match_counts[cluster_number]

    def unweighted_average_self_feature(self, cluster_number, query_features, object_featrues, instance_number):
        self.feature_bank[:, cluster_number] = self.feature_bank[:, cluster_number] + \
            query_features[instance_number]
        self.feature_match_counts[cluster_number] += 1
        self.feature_bank[:,
                          cluster_number] /= 2

        self.object_bank[:, cluster_number] = self.object_bank[:, cluster_number] + \
            object_featrues[instance_number]
        self.object_match_counts[cluster_number] += 1
        self.object_bank[:,
                          cluster_number] /= 2

    def dump_gt_adjacency_matrix(self):
        row_labels = [k for k, _ in sorted(
            self.instance_map.items(), key=lambda item: item[1])]
        mat = render_adj_matrix(self.gt_adjacency_matrix, row_labels)
        sim_mat = Image.fromarray(mat, 'RGB')
        sim_mat.save(
            f'{self.dump_dir}/{self.room_id}_1_{self.steps-1}_adj.png')
        sim_mat.close()

    def dump_features_and_labels(self, query_features, edge_features, labels):
        torch.save(query_features, f'{self.dump_dir}/{self.room_id}_{self.instance_id}_{self.update_count}.pt')
        with open(f'{self.dump_dir}/{self.room_id}_{self.instance_id}_{self.update_count}_label.json', 'w') as f:
            json.dump(labels, f)
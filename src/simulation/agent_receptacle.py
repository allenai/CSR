import json
import os
import random
from itertools import combinations
from src.simulation.constants import CONTROLLER_COMMIT_ID
from string import ascii_letters

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ai2thor.controller import Controller
from PIL import Image
from src.lightning.modules.receptacle_module import ReceptacleModule
from src.shared.constants import (CLASSES_TO_IGNORE, IMAGE_SIZE, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)
from src.shared.utils import check_none_or_empty
from src.simulation.shortest_path_navigator import (AgentLocKeyType,
                                                    ShortestPathNavigatorTHOR)
from torchvision.transforms.transforms import Compose, Normalize, ToTensor


class AgentReceptacle(object):
    def __init__(
        self,
        scene_name='FloorPlan1',
        image_size=IMAGE_SIZE,
        random_start=True,
        trajectory=None,
        model_path=None
    ) -> None:
        super().__init__()
        self.controller = Controller(
            commit_id=CONTROLLER_COMMIT_ID,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=image_size,
            height=image_size
        )

        if random_start and trajectory is not None:
            raise ValueError(
                'cannot set `random_start=True` and also pass a predefined `trajectory`')
        self.model = None
        self.set_scene(scene_name, random_start=random_start,
                       trajectory=trajectory, model_path=model_path)
        self.inference_transform = Compose([
            ToTensor(),
            Normalize(NORMALIZE_RGB_MEAN, NORMALIZE_RGB_STD)
        ])

    def set_scene(self, scene_name, random_start=True, trajectory=None, model_path=None):
        self.controller.reset(scene=scene_name)
        self.spn = ShortestPathNavigatorTHOR(self.controller)
        self.reachable_spots = self.spn.reachable_points_with_rotations_and_horizons()
        self.steps = 0

        if random_start:
            start_index = random.randint(0, len(self.reachable_spots)-1)
            start_key = self.spn.get_key(self.reachable_spots[start_index])
            self.spn.controller.step(
                action='Teleport',
                position=dict(x=start_key[0], y=0.9, z=start_key[1]),
                rotation=dict(x=0, y=start_key[2], z=0),
                horizon=start_key[3],
                standing=True
            )

        self.target_key = None
        self.relations_last_step = None
        self.relationship_memory = {}
        self.trajectory_memory = []
        self.object_position_memory = {}
        self.predefined_trajectory = None

        if trajectory is not None:
            start_key = trajectory[0][0]
            self.spn.controller.step(
                action='Teleport',
                position=dict(x=start_key[0], y=0.9, z=start_key[1]),
                rotation=dict(x=0, y=start_key[2], z=0),
                horizon=start_key[3],
                standing=True
            )

            self.target_key = tuple(trajectory[-1][-1])
            self.predefined_trajectory = trajectory

        if not check_none_or_empty(model_path):
            self.model = ReceptacleModule.load_from_checkpoint(model_path)
            self.model.eval()
            self.model.freeze()

        if self.model is not None:
            self.agg_confidences = {}
            self.relations_last_step_pred = None
            self.relationship_memory_pred = {}

    def set_random_target(self):
        target_index = random.randint(0, len(self.reachable_spots)-1)
        self.target_key = self.spn.get_key(self.reachable_spots[target_index])

        return self.target_key

    def set_target(self, target_key: AgentLocKeyType):
        self.target_key = target_key

    def take_next_action(self):
        if self.target_key is None:
            raise ValueError(
                'self.target_key must be set before we can navigate')

        curr_key = self.spn.get_key(self.spn.last_event.metadata["agent"])
        action = None

        while 1:
            action = None

            if self.predefined_trajectory is not None:
                action = self.predefined_trajectory[self.steps][1]
            else:
                action = self.spn.shortest_path_next_action(
                    curr_key, self.target_key)

            event = self.spn.controller.step(action=action)

            if not event.metadata['lastActionSuccess']:
                if self.predefined_trajectory is None:
                    self.spn.update_graph_with_failed_action(action)
                    continue
                else:
                    raise ValueError(
                        'using predefined trajectory, but action is failing')

            if self.predefined_trajectory is not None:
                assert self.spn.get_key(self.spn.last_event.metadata["agent"]) == tuple(
                    self.predefined_trajectory[self.steps][2])

            self.steps += 1

            self.__update_relations_last_step()
            self.trajectory_memory.append(
                (curr_key, action, self.spn.get_key(self.spn.last_event.metadata["agent"])))

            return event

    def dump_relation_digraph_accumulated(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.relationship_memory, f, indent=4)

    def dump_mask_instances_last_step(self, save_dir, image_prefix):
        for mask_instance in self.relations_last_step:
            mask = self.spn.last_event.instance_masks[mask_instance]
            Image.fromarray(mask).save(os.path.join(
                save_dir, f'{image_prefix}_{mask_instance}.png'), 'PNG')

    def dump_adjacency_gt(self, save_dir, image_prefix):

        if len(self.relationship_memory) == 0:
            return

        sns.set_theme(style="white")

        # Create the pandas DataFrame
        df = nx.to_pandas_adjacency(self.get_gt_nx_digraph(), dtype=int)

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(df, cmap='jet', vmax=1.0, center=0.5,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{image_prefix}.png'))

    def at_goal(self):
        return self.spn.get_key(self.spn.last_event.metadata["agent"]) == self.target_key

    def evaluate_accuracy_vs_confidence(self, threshold):
        if self.model is None:
            raise ValueError('model is None')

        assert len(self.relationship_memory) == len(
            self.relationship_memory_pred)
        thresholds = [0.0, 0.5, 0.75, 0.85, 0.95,
                      0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
        gt_scores = []
        confident_wrong = set()
        unrelated_wrong = set()
        for t in thresholds:
            denom = len(self.relationship_memory)
            if denom == 0:
                return None, None, None, None
            num = 0

            for obj in self.relationship_memory:
                if obj in self.agg_confidences and self.agg_confidences[obj] > t:
                    if self.relationship_memory[obj] == self.relationship_memory_pred[obj]:
                        num += 1
                    elif self.agg_confidences[obj] > threshold:
                        confident_wrong.add(
                            (obj, self.relationship_memory_pred[obj]))
                else:
                    # case where we are implicitly saying there is no relationship for this object
                    if self.relationship_memory[obj] == None:
                        num += 1
                    elif obj in self.agg_confidences and self.agg_confidences[obj] < threshold:
                        unrelated_wrong.add(obj)

            gt_scores.append(float(num) / denom)

        return thresholds, gt_scores, confident_wrong, unrelated_wrong

    def get_gt_nx_digraph(self):
        G = nx.DiGraph(directed=True)

        for src in self.relationship_memory:
            G.add_node(src)
        for src in self.relationship_memory:
            if self.relationship_memory[src] is not None:
                G.add_edge(src, self.relationship_memory[src])

        return G

    def __create_inference_minibatch(self):
        pairs = list(combinations(self.relations_last_step.keys(), 2))

        image = self.inference_transform(
            Image.fromarray(self.spn.last_event.frame))

        # fill the gt for these pairs
        mini_batch_gt = []

        for p in pairs:
            if self.relations_last_step[p[0]] == p[1]:
                mini_batch_gt.append(1)
            elif self.relations_last_step[p[1]] == p[0]:
                mini_batch_gt.append(2)
            else:
                mini_batch_gt.append(0)

        masks = {}
        for mask_instance in self.relations_last_step:
            mask = self.spn.last_event.instance_masks[mask_instance]
            masks[mask_instance] = torch.from_numpy(
                mask).unsqueeze(0)

        input_tensor = torch.empty(
            (len(pairs), 5, image.shape[1], image.shape[2]))
        for i, p in enumerate(pairs):
            input_tensor[i] = torch.cat((image, masks[p[0]], masks[p[1]]), 0)

        return input_tensor, pairs, mini_batch_gt

    def __check_object_visible(self, object_metadata, ignore_classes=CLASSES_TO_IGNORE):
        for c in ignore_classes:
            key = c + '|'
            if key in object_metadata['objectId']:
                return False
        return object_metadata['visible'] and object_metadata['objectId'] in self.spn.last_event.instance_masks

    def __update_relations_last_step(self):
        relations = {}
        for entry in self.spn.last_event.metadata['objects']:
            if self.__check_object_visible(entry):
                self.object_position_memory[entry['objectId']
                                            ] = entry['position']
                if entry['parentReceptacles'] is None:
                    relations[entry['objectId']] = None
                else:
                    for parent_id in entry['parentReceptacles']:
                        for entry2 in self.spn.last_event.metadata['objects']:
                            if entry2['objectId'] == parent_id and \
                                entry['objectId'] in entry2['receptacleObjectIds'] and \
                                    self.__check_object_visible(entry2):

                                relations[entry['objectId']] = parent_id
                                break

                            else:
                                relations[entry['objectId']] = None
        self.relations_last_step = relations
        for r in relations:
            if r not in self.relationship_memory or self.relationship_memory[r] is None:
                self.relationship_memory[r] = relations[r]

        if self.model is not None:
            input_tensor, pairs, _ = self.__create_inference_minibatch()
            self.relations_last_step_pred = {}

            if len(pairs) == 0:
                for r in relations:
                    self.relations_last_step_pred[r] = None
                    if r not in self.relationship_memory_pred:
                        self.relationship_memory_pred[r] = None

                assert len(self.relationship_memory) == len(
                    self.relationship_memory_pred)

                return

            probabilities = torch.softmax(self.model(input_tensor), dim=1)
            conf, preds = torch.max(probabilities, dim=1)

            for obj in self.relations_last_step:
                self.relations_last_step_pred[obj] = None

            for obj in self.relationship_memory:
                if obj not in self.relationship_memory_pred:
                    self.relationship_memory_pred[obj] = None
                if self.relationship_memory[obj] is not None \
                        and self.relationship_memory[obj] not in self.relationship_memory_pred:
                    self.relationship_memory_pred[self.relationship_memory[obj]] = None

            assert len(self.relationship_memory) == len(
                self.relationship_memory_pred)

            for i, p in enumerate(pairs):
                if preds[i].item() == 0:
                    continue

                child = None
                parent = None

                if preds[i].item() == 1:
                    child, parent = p
                if preds[i].item() == 2:
                    parent, child = p

                if child in self.agg_confidences:
                    if conf[i].item() > self.agg_confidences[child]:
                        self.relations_last_step_pred[child] = parent
                        self.agg_confidences[child] = conf[i].item()
                        self.relationship_memory_pred[child] = parent
                else:
                    self.relations_last_step_pred[child] = parent
                    self.agg_confidences[child] = conf[i].item()
                    self.relationship_memory_pred[child] = parent

            assert len(self.relationship_memory) == len(
                self.relationship_memory_pred)

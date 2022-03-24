import json
import os
from copy import deepcopy
from itertools import permutations

import torch
from networkx.algorithms.shortest_paths.weighted import dijkstra_path
from PIL import Image
from scipy.optimize import linear_sum_assignment
from src.shared.utils import render_adj_diff_matrix, render_sim_matrix
from src.simulation.environment import RearrangeTHOREnvironment
from src.simulation.module_box import GtBoxModule
from src.simulation.module_relation_tracking import RelationTrackingModule
from src.simulation.module_state_graph import StateGraphModule
from src.simulation.shortest_path_navigator import ShortestPathNavigatorTHOR
from src.simulation.utils import get_agent_map_data


class PlannerModule(object):
    def __init__(self,
                 env: RearrangeTHOREnvironment,
                 room_id: int,
                 instance_id: int,
                 use_gt_object_matching: bool,
                 dump_dir: str) -> None:
        super().__init__()

        #
        self.use_gt_object_matching = use_gt_object_matching

        #
        self.scene_module_walkthrough = None
        self.state_module_walkthrough = None
        self.scene_module_unshuffle = None
        self.state_module_unshuffle = None
        self.box_stats_walkthrough = None
        self.box_stats_unshuffle = None

        #
        self.env = env
        self.room_id = room_id
        self.instance_id = instance_id
        self.dump_dir = dump_dir
        self.steps = 0

        #
        self.fused_state_module = None
        self.walkthrough_to_fused_map = {}
        self.legs = None
        self.pickup_ids = None

    def store_representations(self, rtm: RelationTrackingModule, sgm: StateGraphModule, bm: GtBoxModule, from_walkthrough: bool):
        if from_walkthrough:
            self.scene_module_walkthrough = deepcopy(rtm)
            self.state_module_walkthrough = deepcopy(sgm)
            self.box_stats_walkthrough = deepcopy(bm.moved_detection_counts)
        else:
            self.scene_module_unshuffle = deepcopy(rtm)
            self.state_module_unshuffle = deepcopy(sgm)
            self.box_stats_unshuffle = deepcopy(bm.moved_detection_counts)

    def generate_plan(self, cos_sim_moved_threshold, cos_sim_object_threshold, debug):

        shared_cluster_id_walkthrough, shared_cluster_id_unshuffle = None, None
        names = None

        if self.use_gt_object_matching:
            shared_cluster_id_walkthrough, shared_cluster_id_unshuffle, names = self._object_match_gt()
        else:
            shared_cluster_id_walkthrough, shared_cluster_id_unshuffle = self._object_match_pred(
                cos_sim_object_threshold)

        shared_cluster_id_walkthrough = torch.tensor(
            shared_cluster_id_walkthrough)
        shared_cluster_id_unshuffle = torch.tensor(shared_cluster_id_unshuffle)

        features_matched_walkthrough = self.scene_module_walkthrough.relationship_bank[
            shared_cluster_id_walkthrough][:, shared_cluster_id_walkthrough, :]
        features_matched_unshuffle = self.scene_module_unshuffle.relationship_bank[
            shared_cluster_id_unshuffle][:, shared_cluster_id_unshuffle, :]
        dotted = torch.einsum(
            "hwc,hwc->hw", features_matched_walkthrough, features_matched_unshuffle)

        # if debug:
        #     adj_walkthrough = torch.norm(features_matched_walkthrough, dim=2)
        #     adj_unshuffle = torch.norm(features_matched_unshuffle, dim=2)
        #     mat = render_sim_matrix(dotted.numpy(), names, names)
        #     img = Image.fromarray(mat, 'RGB')
        #     img.save(f'sim.png')

        #     mat = render_adj_diff_matrix(
        #         adj_walkthrough.numpy(), adj_unshuffle.numpy(), names, names)
        #     img = Image.fromarray(mat, 'RGB')
        #     img.save(f'adj_diff.png')

        # find what moved
        candidate_moved_dotted_id = self._infer_moved(
            dotted, cos_sim_moved_threshold)

        # get the index for the walkthrough state
        _, target_nodes_walkthrough = self._find_cluster_ids_nodes(
            shared_cluster_id_walkthrough, candidate_moved_dotted_id, self.state_module_walkthrough)

        cluster_ids_unshuffle, source_nodes_unshuffle = self._find_cluster_ids_nodes(
            shared_cluster_id_unshuffle, candidate_moved_dotted_id, self.state_module_unshuffle)

        assert len(target_nodes_walkthrough) == len(source_nodes_unshuffle)

        # NOTE: different as we are now taking unshffle as src (1st) not (2nd)
        finals_src_target = []
        finals_cluster_id_unshuffle = []

        for i in range(len(target_nodes_walkthrough)):
            if target_nodes_walkthrough[i] is None and source_nodes_unshuffle[i] is None:
                # good case where something is moveable but we do not detect it has moved
                pass
            elif target_nodes_walkthrough[i] is None or source_nodes_unshuffle[i] is None:
                # bad case where something is moveable but we do have a movable state in both trajectories
                pass
            elif target_nodes_walkthrough[i] is not None and source_nodes_unshuffle[i] is not None:
                # good case where we think something moved and have location for it before and after
                finals_src_target.append(
                    (source_nodes_unshuffle[i], target_nodes_walkthrough[i]))
                finals_cluster_id_unshuffle.append(cluster_ids_unshuffle[i])

        self._fuse_graphs()
        self.legs, self.pickup_ids = self._get_plan(
            finals_src_target, finals_cluster_id_unshuffle)

    def _object_match_gt(self):
        name_to_cluster_id_walkthrough = self._name_to_cluster_id(
            self.scene_module_walkthrough.cluster_meta)
        name_to_cluster_id_unshuffle = self._name_to_cluster_id(
            self.scene_module_unshuffle.cluster_meta)

        shared_objects = name_to_cluster_id_walkthrough.keys(
        ) & name_to_cluster_id_unshuffle.keys()

        shared_cluster_id_walkthrough = []
        shared_cluster_id_unshuffle = []
        names = []
        for k in shared_objects:
            names.append(k)
            shared_cluster_id_walkthrough.append(
                name_to_cluster_id_walkthrough[k])
            shared_cluster_id_unshuffle.append(name_to_cluster_id_unshuffle[k])

        return shared_cluster_id_walkthrough, shared_cluster_id_unshuffle, names

    def _object_match_pred(self, cos_sim_object_threshold):
        sim = torch.transpose(self.scene_module_walkthrough.object_bank, 0, 1)
        sim = torch.matmul(sim, self.scene_module_unshuffle.object_bank)

        # hungarian matching to get the maximal assignment
        w_idx, un_idx = linear_sum_assignment(
            sim.numpy(), maximize=True)

        assert len(w_idx) == len(un_idx)

        w_idx_final = []
        un_idx_final = []
        for i in range(len(w_idx)):
            if (sim[w_idx[i], un_idx[i]] > cos_sim_object_threshold).item():
                w_idx_final.append(w_idx[i])
                un_idx_final.append(un_idx[i])

        return w_idx_final, un_idx_final

    def execute_plan(self, debug):
        assert self.legs is not None

        leg_num = 0
        while len(self.legs) != 0:
            leg = self.legs.pop(0)
            event = self.env.controller.step(
                action='Done', **self.env.physics_step_kwargs)
            if debug:
                self.dump_observation(event)
            if leg_num % 2 == 1:
                pid = self.pickup_ids.pop(0)
                if pid in self.fused_state_module.graph.nodes[leg[0]]['attr']['state'].pickupable_points:
                    x = self.fused_state_module.graph.nodes[leg[0]
                                                            ]['attr']['state'].pickupable_points[pid]['x']
                    y = self.fused_state_module.graph.nodes[leg[0]
                                                            ]['attr']['state'].pickupable_points[pid]['y']
                    self.env.pickup_object(x, y)
                else:
                    pass
                self.steps += 1
                if debug:
                    self.dump_observation(self.env.controller.last_event)
            while len(leg) > 1:
                curr_node = leg.pop(0)
                next_node = leg[0]

                if (curr_node, next_node) not in self.fused_state_module.graph.edges:
                    print('replanning downstream')
                    leg = dijkstra_path(
                        self.fused_state_module.graph, curr_node, leg[-1])
                    tmp = leg.pop(0)
                    assert tmp == curr_node
                    next_node = leg[0]

                # make sure we are not drifting for some reason
                curr_node_key = {
                    'x': self.fused_state_module.graph.nodes[curr_node]['attr']['state'].agent_position['x'],
                    'z': self.fused_state_module.graph.nodes[curr_node]['attr']['state'].agent_position['z'],
                    'rotation': self.fused_state_module.graph.nodes[curr_node]['attr']['state'].agent_rotation["y"],
                    'horizon': self.fused_state_module.graph.nodes[curr_node]['attr']['state'].agent_horizon
                }
                curr_node_key = ShortestPathNavigatorTHOR.get_key(
                    curr_node_key)

                event_key = ShortestPathNavigatorTHOR.get_key(
                    self.env.controller.last_event.metadata['agent'])

                assert curr_node_key == event_key

                action = self.fused_state_module.graph.edges[(
                    curr_node, next_node)]['attr']['action']
                event = None
                if 'Rotate' in action:
                    event = self.env.controller.step(
                        action=action, degrees=90, **self.env.physics_step_kwargs)
                elif 'Look' in action:
                    event = self.env.controller.step(
                        action=action, degrees=30, **self.env.physics_step_kwargs)
                else:
                    event = self.env.controller.step(
                        action=action, **self.env.physics_step_kwargs)

                if not self.env.controller.last_event.metadata["lastActionSuccess"]:
                    # delete edge between two nodes as not traversable

                    # ShortestPathNavigatorTHOR.get_key
                    self.fused_state_module.graph.remove_edge(
                        curr_node, next_node)
                    self.fused_state_module.graph.remove_edge(
                        next_node, curr_node)

                    print(
                        self.env.controller.last_event.metadata["errorMessage"])
                    print('replanning')

                    # NOTE: replan
                    leg = dijkstra_path(
                        self.fused_state_module.graph, curr_node, leg[-1])
                    continue

                self.steps += 1
                if debug:
                    self.dump_observation(event)
            if leg_num % 2 == 1:
                # assert self.env.drop_held_object_with_snap()
                self.env.drop_held_object_with_snap()

                self.steps += 1
                if debug:
                    self.dump_observation(self.env.controller.last_event)

            leg_num += 1

    def dump_observation(self, event):
        im = Image.fromarray(event.frame)
        im.save(
            f'{self.dump_dir}/rearrange_{self.room_id}_{self.instance_id}_{self.steps}.png', 'PNG')
        with open(os.path.join(self.dump_dir, f'rearrange_{self.room_id}_{self.instance_id}_{self.steps}.json'), 'w') as f:
            json.dump(event.metadata, f, indent=4)

        top_down_data = get_agent_map_data(self.env.controller)
        top_img = Image.fromarray(top_down_data['frame'])
        top_img.save(os.path.join(
            self.dump_dir, f'rearrange_{self.room_id}_{self.instance_id}_{self.steps}_top.png'), 'PNG')
        return

    def _infer_moved(self, dotted, cos_sim_moved_threshold):
        candidate_moved_dotted_id = []
        for i in range(dotted.shape[0]):
            if dotted[i, i] < cos_sim_moved_threshold:
                candidate_moved_dotted_id.append(i)

        return candidate_moved_dotted_id

    def _name_to_cluster_id(self, meta):
        name_to_cluster_id = {}
        for entry in meta:
            name_to_cluster_id[meta[entry]['representative']] = int(
                entry)

        return name_to_cluster_id

    def _find_cluster_ids_nodes(self, shared_cluster_id, candidate_moved_dotted_id, state_module):
        possible_move = shared_cluster_id[candidate_moved_dotted_id].tolist()
        target_nodes = []
        for cid in possible_move:
            if cid in state_module.pickupable_cluster_to_node:
                target_nodes.append(
                    state_module.pickupable_cluster_to_node[cid][0])
            else:
                target_nodes.append(None)

        return possible_move, target_nodes

    def _fuse_graphs(self):
        self.fused_state_module = deepcopy(self.state_module_unshuffle)

        # add nodes from walkthrough and keep mapping from old to new node ids
        for walkthrough_node_id in self.state_module_walkthrough.graph.nodes:
            walkthrough_state = self.state_module_walkthrough.graph.nodes[
                walkthrough_node_id]['attr']['state']
            existing_node_id = self.fused_state_module.state_to_node_id(
                walkthrough_state)
            if existing_node_id is None:
                # case where we need to add the a node to the graph
                new_node_id = self.fused_state_module.add_adjoint_node(
                    walkthrough_state)
                self.walkthrough_to_fused_map[walkthrough_node_id] = new_node_id
            else:
                # case where the position has been visited in the unshuffle stage already, keep the ref
                self.walkthrough_to_fused_map[walkthrough_node_id] = existing_node_id

        # add the edges from the walkthrough graph into the fused graph
        for e in self.state_module_walkthrough.graph.edges:
            attr = self.state_module_walkthrough.graph.edges[e]['attr']
            self.fused_state_module.add_adjoint_edge(
                self.walkthrough_to_fused_map[e[0]], self.walkthrough_to_fused_map[e[1]], attr)

    def _get_plan(self, finals_src_target, finals_cluster_id):

        assert len(finals_src_target) == len(finals_cluster_id)

        # init nodes in a graph from unshuffle
        if len(finals_src_target) == 0:
            return [], []

        # greedy plan from current position to src
        best_legs = []
        best_legs_order = None
        best_cost = float('inf')

        # perms = permutations(range(len(finals_src_target)))

        # for p in perms:
        legs = []
        cost = 0
        curr_node = self.fused_state_module.state_to_node_id(
            self.fused_state_module.current_state)
        for src, target in finals_src_target:
            legs.append(dijkstra_path(
                self.fused_state_module.graph, curr_node, src))
            target_ajoint = self.walkthrough_to_fused_map[target]
            legs.append(dijkstra_path(
                self.fused_state_module.graph, src, target_ajoint))
            curr_node = target_ajoint

            # for leg in legs:
            #     cost += len(leg)

            # if cost < best_cost:
            #     best_cost = cost
            #     best_legs = legs.copy()
            #     best_legs_order = list(p)

        # best_cluster_id = [finals_cluster_id[i] for i in best_legs_order]
        best_cluster_id = [finals_cluster_id[i]
                           for i in range(len(finals_cluster_id))]

        # return best_legs, best_cluster_id
        return legs, best_cluster_id

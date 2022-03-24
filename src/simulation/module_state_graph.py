import matplotlib.pyplot as plt
from networkx import draw_networkx
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.classes.digraph import DiGraph
from PIL import ImageChops
from src.simulation.constants import ACTION_NEGATIONS
from src.simulation.state import State


class StateGraphModule(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.graph = DiGraph()

        self.pickupable_cluster_to_node = {}
        self.openable_cluster_to_node = {}
        self.cluster_to_biggest_box_node = {}

        self.node_count = 0
        self.current_state = None

    def find_path(self, src_id, target_id):
        if src_id >= self.node_count:
            raise ValueError(f'src_id not in graph: {src_id}')
        if target_id >= self.node_count:
            raise ValueError(f'target_id not in graph: {target_id}')

        if src_id == target_id:
            return []

        return shortest_path(self.graph, source=src_id, target=target_id)

    def add_adjoint_node(self, state: State):
        node_id = self.node_count
        self.graph.add_node(node_id, attr={'state': state})
        self.node_count += 1

        return node_id

    def add_adjoint_edge(self, node_src, node_dest, attr):
        self.graph.add_edge(node_src, node_dest, attr=attr)

    def add_edge(self, state: State, action: str):
        assert abs(state.agent_rotation['y']) % 90.0 < 0.001

        if action is None:
            # special case where we are adding the first state
            self.graph.add_node(0, attr={'state': state})
            self.node_count = 1
            self.current_state = state

            if len(state.pickupable):
                for cluster_id in state.pickupable:
                    if cluster_id in self.pickupable_cluster_to_node:
                        self.pickupable_cluster_to_node[cluster_id].append(0)
                    else:
                        self.pickupable_cluster_to_node[cluster_id] = [0]

            if len(state.openable):
                for cluster_id in state.openable:
                    if cluster_id in self.openable_cluster_to_node:
                        self.openable_cluster_to_node[cluster_id].append(0)
                    else:
                        self.openable_cluster_to_node[cluster_id] = [0]

            for local_id, cluster_id in enumerate(state.instance_cluster_ids):
                if cluster_id not in self.cluster_to_biggest_box_node:
                    self.cluster_to_biggest_box_node[cluster_id] = (state.boxes[local_id], state.areas[local_id], 0)
                else:
                    if state.areas[local_id] > self.cluster_to_biggest_box_node[cluster_id][1]:
                        self.cluster_to_biggest_box_node[cluster_id] = (state.boxes[local_id], state.areas[local_id], 0)

            return

        assert self.node_count > 0
        assert self.current_state is not None

        if action not in ACTION_NEGATIONS:
            raise ValueError(f'action: {action} not supported')

        node_id = self.state_to_node_id(state)
        if node_id is None:
            node_id = self.node_count
            self.graph.add_node(node_id, attr={'state': state})
            self.node_count += 1

        src = self.state_to_node_id(self.current_state)
        self.graph.add_edge(src, node_id, attr={'action': action})

        negated_action = ACTION_NEGATIONS[action]
        self.graph.add_edge(node_id, src, attr={
                            'action': negated_action})

        self.current_state = state

        if len(state.pickupable):
            for cluster_id in state.pickupable:
                if cluster_id in self.pickupable_cluster_to_node:
                    self.pickupable_cluster_to_node[cluster_id].append(node_id)
                else:
                    self.pickupable_cluster_to_node[cluster_id] = [node_id]

        if len(state.openable):
            for cluster_id in state.openable:
                if cluster_id in self.openable_cluster_to_node:
                    self.openable_cluster_to_node[cluster_id].append(node_id)
                else:
                    self.openable_cluster_to_node[cluster_id] = [node_id]

        for local_id, cluster_id in enumerate(state.instance_cluster_ids):
            if cluster_id not in self.cluster_to_biggest_box_node:
                self.cluster_to_biggest_box_node[cluster_id] = (state.boxes[local_id], state.areas[local_id], node_id)
            else:
                if state.areas[local_id] > self.cluster_to_biggest_box_node[cluster_id][1]:
                    self.cluster_to_biggest_box_node[cluster_id] = (state.boxes[local_id], state.areas[local_id], node_id)

    def state_to_node_id(self, state: State):
        nodes = self.graph.nodes()
        for state_id in nodes:
            existing_state = nodes[state_id]['attr']['state']
            if self.are_same_agent_pose(state, existing_state):
                return state_id

        return None

    def are_same_agent_pose(self, s1: State, s2: State, pos_thres=0.1, rot_thresh=10., hor_thresh=10.):
        keys = ['x', 'y', 'z']
        for k in keys:
            if abs(s1.agent_position[k] - s2.agent_position[k]) > pos_thres:
                return False
            if abs(s1.agent_rotation[k] - s2.agent_rotation[k]) > rot_thresh:
                return False
        if abs(s1.agent_horizon - s2.agent_horizon) > rot_thresh:
            return False

        return True

    def dump_graph(self, from_walkthrough):
        plt.clf()
        color = 'blue'
        if from_walkthrough:
            color = 'green'
        options = {
            'node_color': color,
            'node_size': 200,
            'width': 2,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        draw_networkx(self.graph, arrows=True, **options)
        if from_walkthrough:
            plt.savefig('walkthrough.png')
        else:
            plt.savefig('unshuffle.png')

# based on https://github.com/allenai/ai2thor-rearrangement/blob/main/rearrange/expert.py#L41

import copy
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Sequence,
)

import ai2thor.controller
import ai2thor.server
import networkx as nx
from torch.distributions.utils import lazy_property

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor

from src.simulation.constants import STEP_SIZE

AgentLocKeyType = Tuple[float, float, int, int]


class ShortestPathNavigatorTHOR:
    """Tracks shortest paths in AI2-THOR environments.
    Assumes 90 degree rotations and fixed step sizes.
    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        controller: ai2thor.controller.Controller,
        grid_size: float = STEP_SIZE,
        include_move_left_right: bool = False,
    ):
        """Create a `ShortestPathNavigatorTHOR` instance.
        # Parameters
        controller : An AI2-THOR controller which represents the environment in which shortest paths should be
            computed.
        grid_size : The distance traveled by an AI2-THOR agent when taking a single navigational step.
        include_move_left_right : If `True` the navigational actions will include `MoveLeft` and `MoveRight`, otherwise
            they wil not.
        """
        self._cached_graphs: Dict[str, nx.DiGraph] = {}

        self._current_scene: Optional[nx.DiGraph] = None
        self._current_graph: Optional[nx.DiGraph] = None

        self._grid_size = grid_size
        self.controller = controller

        self._include_move_left_right = include_move_left_right

    @lazy_property
    def nav_actions_set(self) -> frozenset:
        """Navigation actions considered when computing shortest paths."""
        nav_actions = [
            "LookUp",
            "LookDown",
            "RotateLeft",
            "RotateRight",
            "MoveAhead",
        ]
        if self._include_move_left_right:
            nav_actions.extend(["MoveLeft", "MoveRight"])
        return frozenset(nav_actions)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    def on_reset(self):
        """Function that must be called whenever the AI2-THOR controller is
        reset."""
        self._current_scene = None

    @property
    def graph(self) -> nx.DiGraph:
        """A directed graph representing the navigation graph of the current
        scene."""
        if self._current_scene == self.scene_name:
            return self._current_graph

        if self.scene_name not in self._cached_graphs:
            g = nx.DiGraph()
            points = self.reachable_points_with_rotations_and_horizons()
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._cached_graphs[self.scene_name] = g

        self._current_scene = self.scene_name
        self._current_graph = self._cached_graphs[self.scene_name].copy()
        return self._current_graph

    def reachable_points_with_rotations_and_horizons(
        self,
    ) -> List[Dict[str, Union[float, int]]]:
        """Get all the reaachable positions in the scene along with possible
        rotation/horizons."""
        self.controller.step(action="GetReachablePositions")
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in [0, 90, 180, 270]:
            for horizon in [0]:#[-30, 0, 30]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)
        return points

    @staticmethod
    def location_for_key(key, y_value=0.0) -> Dict[str, Union[float, int]]:
        """Return a agent location dictionary given a graph node key."""
        x, z, rot, hor = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot, horizon=hor)
        return loc

    @staticmethod
    def get_key(input_dict: Dict[str, Any], ndigits: int = 2) -> AgentLocKeyType:
        """Return a graph node key given an input agent location dictionary."""
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            rot = input_dict["rotation"]
            hor = input_dict["horizon"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            rot = input_dict["rotation"]["y"]
            hor = input_dict["cameraHorizon"]

        return (
            round(x, ndigits),
            round(z, ndigits),
            round_to_factor(rot, 30) % 360,
            round_to_factor(hor, 30) % 360,
        )

    def update_graph_with_failed_action(self, failed_action: str):
        """If an action failed, update the graph to let it know this happened
        so it won't try again."""
        if (
            self.scene_name not in self._cached_graphs
            or failed_action not in self.nav_actions_set
        ):
            return

        source_key = self.get_key(self.last_event.metadata["agent"])
        self._check_contains_key(source_key)

        edge_dict = self.graph[source_key]
        to_remove_key = None
        for target_key in self.graph[source_key]:
            if edge_dict[target_key]["action"] == failed_action:
                to_remove_key = target_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(
        self, g: nx.DiGraph, s: AgentLocKeyType, t: AgentLocKeyType,
    ):
        """Add an edge to the graph."""

        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot, s_hor = s
        t_x, t_z, t_rot, t_hor = t

        l1_dist = round(abs(s_x - t_x) + abs(s_z - t_z), 2)
        angle_dist = (round_to_factor(t_rot - s_rot, 90) % 360) // 90
        horz_dist = (round_to_factor(t_hor - s_hor, 30) % 360) // 30

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [l1_dist, angle_dist, horz_dist]) != 1:
            return

        grid_size = self._grid_size
        action = None
        if angle_dist != 0:
            if angle_dist == 1:
                action = "RotateRight"
            elif angle_dist == 3:
                action = "RotateLeft"

        elif horz_dist != 0:
            if horz_dist == 11:
                action = "LookUp"
            elif horz_dist == 1:
                action = "LookDown"
        elif ae(l1_dist, grid_size):

            if s_rot == 0:
                forward = round((t_z - s_z) / grid_size)
                right = round((t_x - s_x) / grid_size)
            elif s_rot == 90:
                forward = round((t_x - s_x) / grid_size)
                right = -round((t_z - s_z) / grid_size)
            elif s_rot == 180:
                forward = -round((t_z - s_z) / grid_size)
                right = -round((t_x - s_x) / grid_size)
            elif s_rot == 270:
                forward = -round((t_x - s_x) / grid_size)
                right = round((t_z - s_z) / grid_size)
            else:
                raise NotImplementedError(
                    f"source rotation == {s_rot} unsupported.")

            if forward > 0:
                g.add_edge(s, t, action="MoveAhead")
            elif self._include_move_left_right:
                if forward < 0:
                    # Allowing MoveBack results in some really unintuitive
                    # expert trajectories (i.e. moving backwards to the goal and the
                    # rotating, for now it's disabled.
                    pass  # g.add_edge(s, t, action="MoveBack")
                elif right > 0:
                    g.add_edge(s, t, action="MoveRight")
                elif right < 0:
                    g.add_edge(s, t, action="MoveLeft")

        if action is not None:
            g.add_edge(s, t, action=action)

    @lazy_property
    def possible_neighbor_offsets(self) -> Tuple[AgentLocKeyType, ...]:
        """Offsets used to generate potential neighbors of a node."""
        grid_size = round(self._grid_size, 2)
        offsets = []
        for rot_diff in [-90, 0, 90]:
            for horz_diff in [-30, 0, 30, 60]:
                for x_diff in [-grid_size, 0, grid_size]:
                    for z_diff in [-grid_size, 0, grid_size]:
                        if (rot_diff != 0) + (horz_diff != 0) + (x_diff != 0) + (
                            z_diff != 0
                        ) == 1:
                            offsets.append(
                                (x_diff, z_diff, rot_diff, horz_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: AgentLocKeyType):
        """Add a node to the graph along with any adjacent edges."""
        if s in graph:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for x_diff, z_diff, rot_diff, horz_diff in self.possible_neighbor_offsets:
            t = (
                s[0] + x_diff,
                s[1] + z_diff,
                (s[2] + rot_diff) % 360,
                (s[3] + horz_diff) % 360,
            )
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    def _check_contains_key(self, key: AgentLocKeyType, add_if_not=True) -> bool:
        """Check if a node key is in the graph.
        # Parameters
        key : The key to check.
        add_if_not : If the key doesn't exist and this is `True`, the key will be added along with
            edges to any adjacent nodes.
        """
        key_in_graph = key in self.graph
        if not key_in_graph:
            get_logger().debug(
                "{} was not in the graph for scene {}.".format(
                    key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)
                if key not in self._cached_graphs[self.scene_name]:
                    self._add_node_to_graph(
                        self._cached_graphs[self.scene_name], key)
        return key_in_graph

    def shortest_state_path(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ) -> Optional[Sequence[AgentLocKeyType]]:
        """Get the shortest path between node keys."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        path = nx.shortest_path(
            G=self.graph, source=source_state_key, target=goal_state_key
        )
        return path

    def action_transitioning_between_keys(self, s: AgentLocKeyType, t: AgentLocKeyType):
        """Get the action that takes the agent from node s to node t."""
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next node key on the shortest path from the source to the
        goal."""
        if source_state_key == goal_state_key:
            raise RuntimeError(
                "called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next action along the shortest path from the source to the
        goal."""
        next_state_key = self.shortest_path_next_state(
            source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_path_next_action_multi_target(
        self,
        source_state_key: AgentLocKeyType,
        goal_state_keys: Sequence[AgentLocKeyType],
    ):
        """Get the next action along the shortest path from the source to the
        closest goal."""
        self._check_contains_key(source_state_key)

        terminal_node = (-1.0, -1.0, -1, -1)
        self.graph.add_node(terminal_node)
        for gsk in goal_state_keys:
            self._check_contains_key(gsk)
            self.graph.add_edge(gsk, terminal_node, action=None)

        next_state_key = self.shortest_path_next_state(
            source_state_key, terminal_node)
        action = self.graph.get_edge_data(
            source_state_key, next_state_key)["action"]

        self.graph.remove_node(terminal_node)
        return action

    def shortest_path_length(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the path shorest path length between the source and the goal."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")

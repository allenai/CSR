from dataclasses import dataclass
from typing import Dict, List

from PIL import Image


@dataclass
class State:
    instance_cluster_ids: List = None
    # boxes: None
    # instance_names: None
    pickupable_points: Dict = None
    openable_points: Dict = None
    pickupable: List = None
    openable: List = None
    boxes: List = None
    areas: Dict = None
    image: Image = None
    agent_position: Dict = None
    agent_rotation: Dict = None
    agent_horizon: float = None
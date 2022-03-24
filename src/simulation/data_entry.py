from dataclasses import dataclass
import hashlib
from typing import Dict, List
from src.simulation.shortest_path_navigator import AgentLocKeyType


@dataclass
class DataEntry(object):
    first_name: str = ''
    second_name: str = ''
    receptacle: int = 0
    receptacle_sibling: int = 0
    room_id: int = 0
    trajectory_id: int = 0
    timestep: int = 0
    position: Dict = None
    rotation: Dict = None
    horizon: Dict = None
    objects_relative_distance: float = -1.
    in_frame_negatives: List = None
    has_shuffle_negatives: bool = False

    @property
    def get_instance_key(self) -> str:
        key_str = f'{self.first_name},{self.second_name}'

        return hashlib.md5(key_str.encode()).hexdigest()

    @property
    def get_category_key(self) -> str:
        first = self.first_name.split('_')[0]
        second = self.second_name.split('_')[0]
        key_str = f'{first},{second}'

        return hashlib.md5(key_str.encode()).hexdigest()

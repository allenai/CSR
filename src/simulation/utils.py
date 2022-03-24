import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
from ai2thor.controller import Controller
import torch
from src.shared.constants import CLASSES_TO_IGNORE
from src.simulation.constants import \
    OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES


def valid_box_size(event, object_id, box_frac_threshold):
    top = (event.instance_detections2D[object_id][0],
           event.instance_detections2D[object_id][1])
    bottom = (event.instance_detections2D[object_id][2] - 1,
              event.instance_detections2D[object_id][3] - 1)

    area = (bottom[0] - top[0]) * (bottom[1] - top[1])

    if area / (event.metadata["screenWidth"] * event.metadata["screenHeight"]) < box_frac_threshold:
        return False

    return True

def compute_iou(pred, target):
    with torch.no_grad():
        assert pred.shape == target.shape

        intersection = target & pred
        union = target | pred
        iou = torch.sum(intersection).flatten() / torch.sum(union).float()

        return iou

def are_images_near(image1, image2, max_mean_pixel_diff=0.5):
    return np.mean(np.abs(image1 - image2).flatten()) <= max_mean_pixel_diff


def are_images_far(image1, image2, min_mean_pixel_diff=10):
    return np.mean(np.abs(image1 - image2).flatten()) >= min_mean_pixel_diff


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array(
            (cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def get_pickupable_objects(event, ignore_classes=CLASSES_TO_IGNORE, distance_thresh=1.5):
    objects_metadata = event.metadata['objects']

    names = []

    for object_metadata in objects_metadata:
        if object_metadata['objectType'] in CLASSES_TO_IGNORE:
            continue

        if not object_metadata['visible']:
            continue

        if object_metadata['distance'] > distance_thresh:
            continue

        if object_metadata['pickupable']:
            names.append(object_metadata['name'])

    return names


def get_openable_objects(event, ignore_classes=CLASSES_TO_IGNORE, distance_thresh=1.5):
    objects_metadata = event.metadata['objects']

    names = []

    for object_metadata in objects_metadata:
        if object_metadata['objectType'] in CLASSES_TO_IGNORE:
            continue

        if not object_metadata['visible']:
            continue

        if object_metadata['distance'] > distance_thresh:
            continue

        if object_metadata['openable']:
            names.append(object_metadata['name'])

    return names


def get_interactable_objects(event, ignore_classes=CLASSES_TO_IGNORE, distance_thresh=1.5):
    objects_metadata = event.metadata['objects']

    ret = []

    for object_metadata in objects_metadata:
        if object_metadata['objectType'] in CLASSES_TO_IGNORE:
            continue

        if not object_metadata['visible']:
            continue

        if object_metadata['distance'] > distance_thresh:
            continue

        if object_metadata['pickupable'] or object_metadata['openable']:
            ret.append(object_metadata['name'])

    return ret


def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])


def get_agent_map_data(c: Controller):
    c.step({"action": "ToggleMapView"})
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        c.last_event.frame.shape, position_to_tuple(
            cam_position), cam_orth_size
    )
    to_return = {
        "frame": c.last_event.frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    c.step({"action": "ToggleMapView"})
    return to_return


def open_objs(
    objects_to_open: List[Dict[str, Any]], controller: Controller
) -> Dict[str, Optional[float]]:
    """Opens up the chosen pickupable objects if they're openable."""
    out: Dict[str, Optional[float]] = defaultdict(lambda: None)
    for obj in objects_to_open:
        last_openness = obj["openness"]
        new_openness = last_openness
        while abs(last_openness - new_openness) <= 0.2:
            new_openness = random.random()

        controller.step(
            "OpenObject",
            objectId=obj["objectId"],
            openness=new_openness,
            forceAction=True,
        )
        out[obj["name"]] = new_openness
    return out


def get_object_ids_to_not_move_from_object_types(
    controller: Controller, object_types: Set[str]
) -> List[str]:
    object_types = set(object_types)
    return [
        o["objectId"]
        for o in controller.last_event.metadata["objects"]
        if o["objectType"] in object_types
    ]


def remove_objects_until_all_have_identical_meshes(controller: Controller):
    obj_type_to_obj_list = defaultdict(lambda: [])
    for obj in controller.last_event.metadata["objects"]:
        obj_type_to_obj_list[obj["objectType"]].append(obj)

    for obj_type in OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES:
        objs_of_type = list(
            sorted(obj_type_to_obj_list[obj_type], key=lambda x: x["name"])
        )
        random.shuffle(objs_of_type)
        objs_to_remove = objs_of_type[:-1]
        for obj_to_remove in objs_to_remove:
            obj_to_remove_name = obj_to_remove["name"]
            obj_id_to_remove = next(
                obj["objectId"]
                for obj in controller.last_event.metadata["objects"]
                if obj["name"] == obj_to_remove_name
            )
            controller.step("RemoveFromScene", objectId=obj_id_to_remove)
            if not controller.last_event.metadata["lastActionSuccess"]:
                return False
    return True

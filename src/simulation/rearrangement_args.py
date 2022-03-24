from dataclasses import dataclass


@dataclass
class RearrangementArgs(object):
    instance_id: int = -1
    room_id: int = -1
    num_steps: int = 250,
    render_instance_segmentation: bool = False
    visibility_distance: float = 20.0
    rotation_degrees: int = 30
    box_frac_threshold: float = 0.008
    box_conf_threshold: float = 0.5
    cos_sim_match_threshold: float = 0.5
    cos_sim_object_threshold: float = 0.5
    cos_sim_moved_threshold: float = 0.65
    averaging_strategy: str = 'weighted'
    gt_exploration_strategy: str = 'waypoint'

    # inference models
    boxes_model_path: str = ''
    boxes_model_type: str = 'maskrcnn'
    exploration_model_path: str = ''
    exploration_cache_dir: str = ''
    are_close_model_path: str = ''
    relation_tracking_model_path: str = ''
    object_tracking_model_path: str = ''
    device_relation_tracking: int = -1 # -1 indicates cpu

    # flags to toggle using gt
    use_gt_boxes: bool = True
    use_gt_exploration: bool = True
    use_gt_are_close: bool = True
    use_gt_relation_tracking: bool = True
    use_gt_object_matching: bool = True
    use_roi_feature_within_traj: bool = False
    use_roi_feature_between_traj: bool = False
    use_box_within_traj: bool = False

    # debug options
    debug: bool = False
    dump_dir: str = './tmp'

    roomr_dir: str = ''
    roomr_meta_dir: str = ''
    data_split: str = 'val'


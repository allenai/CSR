import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from PIL import Image, ImageDraw
from src.shared.constants import CLASSES_TO_IGNORE, IMAGE_SIZE
from src.shared.utils import get_device
from src.simulation.constants import OMNI_CATEGORIES, OMNI_TO_ITHOR
from src.simulation.utils import compute_iou
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.ops import nms


class GtBoxModule(object):
    def __init__(self, box_conf_threshold, box_frac_threshold, model_type, model_path, device_num, moved_detection_counts, get_roi_features, debug) -> None:
        super().__init__()
        self.model_types = {'alfred', 'ithor',
                            'retinanet', 'maskrcnn', 'lvis', 'rpn'}
        if model_type not in self.model_types:
            raise ValueError('Unsupported model type')

        self.transform = T.Compose([T.ToTensor()])
        self.debug = debug
        self.box_conf_threshold = box_conf_threshold
        self.box_frac_threshold = box_frac_threshold
        self.model = None
        self.model_type = model_type
        self.moved_detection_counts = moved_detection_counts
        self.get_roi_features = get_roi_features
        if get_roi_features:
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml'))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1235
            cfg.MODEL.WEIGHTS = model_path
            cfg.INPUT.MIN_SIZE_TEST = 300
            self.cfg = cfg
            self.model = DefaultPredictor(cfg)

    def reset(self):
        for o in self.moved_detection_counts:
            self.moved_detection_counts[o]['count'] = 0

    def get_boxes(self, event):
        step_instances = []
        boxes = {}
        interaction_points = {}
        areas = {}

        count = 0
        boxes_for_detectron = []

        for o in event.metadata['objects']:
            object_id = o['objectId']
            object_name = o['name']
            if event.instance_detections2D is not None and object_id in event.instance_detections2D and o['visible']:
                if o['objectType'] in CLASSES_TO_IGNORE:
                    continue

                top = (event.instance_detections2D[object_id][0],
                       event.instance_detections2D[object_id][1])
                bottom = (event.instance_detections2D[object_id][2] - 1,
                          event.instance_detections2D[object_id][3] - 1)

                area = (bottom[0] - top[0]) * (bottom[1] - top[1])

                if area / (IMAGE_SIZE * IMAGE_SIZE) < self.box_frac_threshold:
                    continue

                step_instances.append(object_name)
                boxes_for_detectron.append(event.instance_detections2D[object_id])

                box = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
                tmp = ImageDraw.Draw(box)
                tmp.rectangle([top, bottom], fill="white")
                trans = T.ToTensor()
                boxes[count] = trans(box)

                mask_idx = event.instance_masks[object_id].nonzero()

                idx = np.random.choice(range(mask_idx[0].shape[0]))
                y = float(mask_idx[0][idx]) / IMAGE_SIZE
                x = float(mask_idx[1][idx]) / IMAGE_SIZE

                interaction_points[count] = {'x': x, 'y': y}
                areas[count] = area
                count += 1

                assert count == len(step_instances)

        feats = None
        if self.get_roi_features:
            img = event.frame.copy()
            inputs = [{"image": torch.as_tensor(img.astype("float32")).permute(2, 0, 1), "height": 224, "width": 224}]
            with torch.no_grad():
                images = self.model.model.preprocess_image(inputs)  # don't forget to preprocess
                features = self.model.model.backbone(images.tensor)  # set of cnn features
                proposals, _ = self.model.model.proposal_generator(images, features, None)  # RPN

                dev = proposals[0].proposal_boxes.tensor.get_device()
                proposals[0].proposal_boxes.tensor = torch.tensor(boxes_for_detectron).float().to(dev)
                features_ = [features[f] for f in self.model.model.roi_heads.box_in_features]
                box_features = self.model.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                feats = self.model.model.roi_heads.box_head(box_features)  # features of all 1k candidates

        if feats is not None:
            assert feats.shape[0] == len(step_instances)

        return step_instances, boxes, interaction_points, areas, feats


class PredBoxModule(object):
    def __init__(self, box_conf_threshold, box_frac_threshold, model_type, model_path, device_num, moved_detection_counts, get_roi_features, debug) -> None:
        super().__init__()
        self.model_types = {'alfred', 'ithor',
                            'retinanet', 'maskrcnn', 'lvis', 'rpn'}
        if model_type not in self.model_types:
            raise ValueError('Unsupported model type')

        self.transform = T.Compose([T.ToTensor()])
        self.debug = debug
        self.box_conf_threshold = box_conf_threshold
        self.box_frac_threshold = box_frac_threshold
        self.model = None
        self.model_type = model_type
        self.moved_detection_counts = moved_detection_counts
        self.get_roi_features = get_roi_features
        self._init_model(model_type, model_path,
                         box_conf_threshold, device_num)

    def reset(self):
        for o in self.moved_detection_counts:
            self.moved_detection_counts[o]['count'] = 0

    def get_boxes(self, event):
        # get the pred boxes

        boxes = None

        img = event.frame.copy()

        feats = None
        if self.get_roi_features:
            img = event.frame.copy()
            inputs = [{"image": torch.as_tensor(img.astype("float32")).permute(2, 0, 1), "height": 224, "width": 224}]
            with torch.no_grad():
                images = self.model.model.preprocess_image(inputs)  # don't forget to preprocess
                features = self.model.model.backbone(images.tensor)  # set of cnn features
                proposals, _ = self.model.model.proposal_generator(images, features, None)  # RPN

                features_ = [features[f] for f in self.model.model.roi_heads.box_in_features]
                box_features = self.model.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                box_features = self.model.model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = self.model.model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = self.model.model.roi_heads.box_predictor.inference(predictions, proposals)
                pred_instances = self.model.model.roi_heads.forward_with_given_boxes(features, pred_instances)

                # output boxes, masks, scores, etc
                pred_instances = self.model.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]
                boxes = pred_instances[0]['instances'].pred_boxes
                ithor_idx = []
                tmp = []
                for i in range(len(pred_instances[0]['instances'])):
                    omni_cat = OMNI_CATEGORIES[pred_instances[0]['instances'][i].pred_classes]
                    if omni_cat in OMNI_TO_ITHOR:
                        ithor_idx.append(i)
                        tmp.append(omni_cat)
                boxes = boxes[ithor_idx]
                feats = feats[ithor_idx]

        else:
            if self.model_type == 'ithor' or self.model_type == 'retinanet' or self.model_type == 'maskrcnn' or self.model_type == 'lvis':
                outputs = self.model(img)
                boxes = outputs['instances'].pred_boxes
                ithor_idx = []
                tmp = []
                if self.model_type == 'ithor':
                    for i in range(len(outputs['instances'])):
                        omni_cat = OMNI_CATEGORIES[outputs['instances'][i].pred_classes]
                        if omni_cat in OMNI_TO_ITHOR:
                            ithor_idx.append(i)
                            tmp.append(omni_cat)
                    boxes = boxes[ithor_idx]

            elif self.model_type == 'rpn':
                outputs = self.model(img)
                idx = torch.sigmoid(
                    outputs['proposals'].objectness_logits) > self.box_conf_threshold
                boxes = outputs['proposals'][idx].proposal_boxes

        gt_step_instances = []
        gt_boxes = {}
        gt_interaction_points = {}

        pred_boxes = {}
        pred_interaction_points = {}
        pred_areas = {}

        count = 0
        feature_idx = []
        for i in range(len(boxes)):
            box = boxes[i].tensor[0]

            top = (box[0], box[1])
            bottom = (box[2], box[3])

            area = (bottom[0] - top[0]) * (bottom[1] - top[1])

            if area / (IMAGE_SIZE * IMAGE_SIZE) < self.box_frac_threshold:
                continue
            if self.model_type == 'ithor':
                pass

            if feats is not None:
                feature_idx.append(i)

            box = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
            tmp = ImageDraw.Draw(box)
            tmp.rectangle([top, bottom], fill="white")
            trans = T.ToTensor()
            pred_boxes[count] = trans(box)
            pred_areas[count] = area
            count += 1

        for o in event.metadata['objects']:
            object_id = o['objectId']
            object_name = o['name']
            if event.instance_detections2D is not None and object_id in event.instance_detections2D and o['visible']:
                if o['objectType'] in CLASSES_TO_IGNORE:
                    continue

                top = (event.instance_detections2D[object_id][0],
                       event.instance_detections2D[object_id][1])
                bottom = (event.instance_detections2D[object_id][2] - 1,
                          event.instance_detections2D[object_id][3] - 1)

                area = (bottom[0] - top[0]) * (bottom[1] - top[1])

                # if area / (IMAGE_SIZE * IMAGE_SIZE) < self.box_frac_threshold:
                #     continue

                gt_step_instances.append(object_name)

                box = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE))
                tmp = ImageDraw.Draw(box)
                tmp.rectangle([top, bottom], fill="white")
                trans = T.ToTensor()
                gt_boxes[object_name] = trans(box)

                mask_idx = event.instance_masks[object_id].nonzero()

                idx = np.random.choice(range(mask_idx[0].shape[0]))
                y = float(mask_idx[0][idx]) / IMAGE_SIZE
                x = float(mask_idx[1][idx]) / IMAGE_SIZE

                gt_interaction_points[object_name] = {'x': x, 'y': y}

        # NOTE: implementation detail for finding a control point for each pred detection
        # to do this we look for overlap with GT, this is used to implement picking up
        # by object box
        class_maps = {}
        step_instances = []  # NOTE: keep track at the matching gt box name for metrics
        for pred_box_id in pred_boxes:
            max_iou = 0.0
            class_maps[pred_box_id] = None
            pred_interaction_points[pred_box_id] = {'x': 0.0, 'y': 0.0}
            step_instance = 'None'
            for gt_box_id in gt_boxes:
                computed_iou = compute_iou(
                    pred_boxes[pred_box_id].long(), gt_boxes[gt_box_id].long()).item()
                if computed_iou > max_iou:
                    max_iou = computed_iou
                    pred_interaction_points[pred_box_id] = gt_interaction_points[gt_box_id]
                    class_maps[pred_box_id] = gt_box_id
                    step_instance = gt_box_id
            step_instances.append(step_instance)

        for i in step_instances:
            if i in self.moved_detection_counts:
                self.moved_detection_counts[i]['count'] += 1

        if feats is not None:
            feats = feats[feature_idx]
            assert feats.shape[0] == len(step_instances)

        return step_instances, pred_boxes, pred_interaction_points, pred_areas, feats

    def _init_model(self, model_type: str, model_path, box_conf_threshold: float, device_num: int):
        if model_type == 'alfred':
            self.model = maskrcnn_resnet50_fpn(num_classes=119)
            d = torch.load(model_path, map_location=get_device(device_num))
            self.model.load_state_dict(d)
            self.model.eval()
        elif model_type == 'ithor':
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml'))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1235
            cfg.MODEL.WEIGHTS = model_path
            cfg.INPUT.MIN_SIZE_TEST = 300
            self.cfg = cfg
            self.model = DefaultPredictor(cfg)
        elif model_type == 'retinanet':
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = box_conf_threshold
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/retinanet_R_50_FPN_3x.yaml")
            cfg.INPUT.FORMAT = 'RGB'
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            self.cfg = cfg
            self.model = DefaultPredictor(cfg)
        elif model_type == 'maskrcnn':
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.INPUT.FORMAT = 'RGB'
            self.cfg = cfg
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            self.model = DefaultPredictor(cfg)
        elif model_type == 'lvis':
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = box_conf_threshold
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
            cfg.INPUT.FORMAT = 'RGB'
            self.cfg = cfg
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            self.model = DefaultPredictor(cfg)
        elif model_type == 'rpn':
            setup_logger()
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/rpn_R_50_FPN_1x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/rpn_R_50_FPN_1x.yaml")
            cfg.INPUT.FORMAT = 'RGB'

            # low threshold means more pruning
            cfg.MODEL.RPN.NMS_THRESH = 0.01

            self.cfg = cfg
            if device_num < 0:
                cfg.MODEL.DEVICE = 'cpu'
            self.model = DefaultPredictor(cfg)
        else:
            raise ValueError('Unsupported model type')

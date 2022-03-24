import io
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from pytorch_lightning.utilities.seed import seed_everything
from src.shared.constants import (IMAGE_SIZE, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image


def check_none_or_empty(input):
    return input is None or input == ''


def count_learnable_parameters(module):
    return sum(p.numel() for p in module.parameters())


def next_power_eight(x):
    # from: https://www.geeksforgeeks.org/round-to-next-greater-multiple-of-8/
    return ((x + 7) & (-8))


def render_confusion_matrix(conf_mat: np.ndarray, class_names: List[str]) -> np.ndarray:
    # based on: https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning

    df_cm = pd.DataFrame(
        conf_mat.astype(np.int64),
        index=np.arange(conf_mat.shape[0]),
        columns=class_names)
    plt.figure()
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    return np.asarray(im, dtype=np.uint8)


def render_sim_matrix(conf_mat: np.ndarray, rows: List[str], cols: List[str], vmin: int = -1, vmax: int = 1) -> np.ndarray:
    # based on: https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning

    df_cm = pd.DataFrame(
        conf_mat.astype(np.float32),
        index=rows,  # np.arange(conf_mat.shape[0]),
        columns=cols)
    plt.figure()
    plt.subplots(figsize=(30, 30))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={
               "size": 20}, fmt='.2f', vmin=vmin, vmax=vmax, cmap='jet')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    return np.asarray(im, dtype=np.uint8)


def render_adj_diff_matrix(mat1: np.ndarray, mat2: np.ndarray, rows: List[str], cols: List[str]) -> np.ndarray:
    # based on: https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning

    mat = np.zeros_like(mat1)

    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            if mat1[i, j] < 0.5 and mat2[i, j] < 0.5:
                mat[i, j] = 0.0
            if mat1[i, j] < 0.5 and mat2[i, j] > 0.5:
                mat[i, j] = 0.33
            if mat1[i, j] > 0.5 and mat2[i, j] < 0.5:
                mat[i, j] = 0.66
            if mat1[i, j] > 0.5 and mat2[i, j] > 0.5:
                mat[i, j] = 1.0

    df_cm = pd.DataFrame(
        mat.astype(np.float32),
        index=rows,  # np.arange(conf_mat.shape[0]),
        columns=cols)
    plt.figure()
    plt.subplots(figsize=(30, 30))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=False, vmin=0, vmax=1, cmap='jet')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    return np.asarray(im, dtype=np.uint8)


def render_adj_matrix(adj_mat: np.ndarray, rows: List[str]) -> np.ndarray:
    # based on: https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning

    df_cm = pd.DataFrame(
        adj_mat.astype(np.int8),
        index=rows,  # np.arange(conf_mat.shape[0]),
        columns=rows)
    plt.figure()
    plt.subplots(figsize=(30, 30))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={
               "size": 20}, vmin=0, vmax=1, cmap='jet')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    return np.asarray(im, dtype=np.uint8)


def render_receptacle_matrix(mat: np.ndarray, rows: List[str]) -> np.ndarray:
    # based on: https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning

    df_cm = pd.DataFrame(
        adj_mat.astype(np.int8),
        index=rows,  # np.arange(conf_mat.shape[0]),
        columns=rows)
    plt.figure()
    plt.subplots(figsize=(30, 30))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={
               "size": 20}, vmin=0, vmax=2, cmap='jet')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)

    return np.asarray(im, dtype=np.uint8)


def reverse_dictonary(d):
    d_inv = {}
    for k, v in d.items():  # <- missing ()
        d_inv[v] = d_inv.get(v, [])
        d_inv[v].append(k)

    return d_inv


def compute_3d_dist(p1, p2):
    p1_np = np.array([p1['x'], p1['y'], p1['z']])
    p2_np = np.array([p2['x'], p2['y'], p2['z']])

    squared_dist = np.sum((p1_np-p2_np)**2, axis=0)
    return np.sqrt(squared_dist)

def get_device(device_number):
    if device_number >= 0:# and torch.cuda.is_available():
        device = torch.device("cuda:{0}".format(device_number))
    else:
        device = torch.device("cpu")

    return device

def load_lightning_inference(checkpoint_path, module_class):
    model = module_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    return model


def load_lightning_train(checkpoint_path, module_class):
    model = module_class.load_from_checkpoint(checkpoint_path)

    return model


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    if torch_seed + worker_id >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)
    random.seed(torch_seed + worker_id)


def get_box(corners, random_box=False):
    if random_box and corners is None:
        t_min, t_max = random.randint(IMAGE_SIZE), random.randint(IMAGE_SIZE)
        x_min, x_max = min(t_min, t_max), max(t_min, t_max)
        t_min, t_max = random.randint(IMAGE_SIZE), random.randint(IMAGE_SIZE)
        y_min, y_max = min(t_min, t_max), max(t_min, t_max)

        corners = [[x_min, x_max], [y_min, y_max]]

    box = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    box[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]] = 1.

    return box.unsqueeze(0)


def dump_batch(relation_query, relation_key, dump_dir, batch_count):
    b = relation_query['image'].shape[0]
    h = relation_query['image'].shape[2]
    w = relation_query['image'].shape[3]

    std = torch.tensor(NORMALIZE_RGB_STD).unsqueeze(
        -1).unsqueeze(-1).repeat(1, h, w)
    mean = torch.tensor(NORMALIZE_RGB_STD).unsqueeze(
        -1).unsqueeze(-1).repeat(1, h, w)

    q_objs = (relation_query['image'].cpu() * std + mean)  # * torch.clamp(
    # relation_query['mask_1'].cpu() + relation_query['mask_2'].cpu(), 0, 1)
    k_objs = (relation_key['image'].cpu() * std + mean)  # * torch.clamp(
    # relation_key['mask_1'].cpu() + relation_key['mask_2'].cpu(), 0, 1)
    # * torch.clamp(
    s_objs = (relation_key['shuffle_image'].cpu() * std + mean)
    # relation_key['shuffle_mask_1'].cpu() + relation_key['shuffle_mask_2'].cpu(), 0, 1)

    for i in range(b):
        if relation_key['has_shuffle_negative'][i]:
            save_image(q_objs[i], os.path.join(dump_dir, f'{batch_count}_{i}_query.png'))
            save_image(k_objs[i], os.path.join(dump_dir, f'{batch_count}_{i}_key.png'))
            save_image(s_objs[i], os.path.join(dump_dir, f'{batch_count}_{i}_shuffle.png'))


def my_shuffle_evaluate(encoder_q, relation1, relation2, device, dump_path, self_feature_only, relational_feature_only, batch_count):

    if dump_path is not None and os.path.exists(dump_path):
        dump_batch(relation1, relation2, dump_path, batch_count)
    # exit(0)

    query = torch.cat(
        (relation1['image'], relation1['mask_1'], relation1['mask_2']), 1).to(device)
    shuffle_negative = torch.cat(
        (relation2['shuffle_image'], relation2['shuffle_mask_1'], relation2['shuffle_mask_2']), 1).to(device)
    positive = torch.cat(
        (relation2['image'], relation2['mask_1'], relation2['mask_2']), 1).to(device)

    has_negatives = relation2['has_shuffle_negative'] > 0.5

    other_mask = torch.ones_like(has_negatives).bool()
    if self_feature_only:
        other_mask = relation1['self'] > 0.5
    elif relational_feature_only:
        other_mask = relation1['self'] < 0.5

    has_negatives = has_negatives & other_mask

    e_q = encoder_q(query)
    e_q = nn.functional.normalize(e_q, dim=1)

    e_n = encoder_q(shuffle_negative)
    e_n = nn.functional.normalize(e_n, dim=1)

    e_k = encoder_q(positive)
    e_k = nn.functional.normalize(e_k, dim=1)

    l_pos = torch.einsum('nc,nc->n', [e_q, e_k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,nc->n', [e_q, e_n]).unsqueeze(-1)

    if torch.any(has_negatives):
        logits = torch.cat((l_pos, l_neg), 1)[has_negatives] / 0.07

        decisions = torch.max(logits, dim=1)
        misses = torch.sum(decisions.indices)
        total = decisions.indices.shape[0]
        loss_shuffle = F.cross_entropy(logits.float(), torch.zeros(
            logits.shape[0]).long().to(device))

        return misses, total, loss_shuffle, logits

    return None, None, None, None

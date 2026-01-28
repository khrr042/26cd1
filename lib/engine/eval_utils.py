import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lib.models import build_model
from lib.utils.post_processing import comput_distmat, rerank_gpu


def load_model(cfg, checkpoint_path, device, num_classes):
    model = build_model(cfg, num_classes)
    if model is None:
        raise ValueError("build_model returned None")
    model = model.to(device)
    model.eval()

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model


def extract_features(model, loader, device):
    feats, pids, camids, img_paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img, pid, camid, paths = batch
            img = img.to(device)
            feat = model(img)
            if isinstance(feat, tuple):
                feat = feat[0]
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))
            img_paths.extend(list(paths))

    feats = torch.cat(feats, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)
    return feats, pids, camids, img_paths


def l2_normalize(feats):
    return F.normalize(feats, dim=1, p=2)


def cosine_distmat(q_feats, g_feats):
    return (1 - torch.mm(q_feats, g_feats.t())).numpy()


def euclidean_distmat(q_feats, g_feats):
    return comput_distmat(q_feats, g_feats).numpy()


def rerank_distmat(q_feats, g_feats, device):
    return rerank_gpu(q_feats.to(device), g_feats.to(device))


def load_tracklets(track_path) -> List[List[str]]:
    if not track_path:
        return []
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Tracklets file not found: {track_path}")
    if track_path.lower().endswith(".json"):
        with open(track_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("tracks", [])
            return data
    tracks = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tracks.append([p.strip() for p in line.split(",") if p.strip()])
    return tracks


def apply_tracklet_min_dist(distmat: np.ndarray, gallery_paths: List[str], tracks: List[List[str]]):
    if not tracks:
        return distmat
    name_to_idx = {os.path.basename(p): i for i, p in enumerate(gallery_paths)}
    for track in tracks:
        idxs = [name_to_idx[n] for n in track if n in name_to_idx]
        if not idxs:
            continue
        min_dist = distmat[:, idxs].min(axis=1)
        distmat[:, idxs] = min_dist[:, None]
    return distmat

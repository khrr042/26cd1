import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from lib.config import cfg
from lib.data import make_data_loader
from lib.utils.metrics import eval_func
from lib.utils.utils import setup_logger
from lib.engine.eval_utils import (
    apply_tracklet_min_dist,
    cosine_distmat,
    extract_features,
    l2_normalize,
    load_model,
    load_tracklets,
    rerank_distmat,
)

def _save_result(output_path, result):
    if output_path is None:
        return
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if output_path.lower().endswith(".jsonl"):
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    elif output_path.lower().endswith(".csv"):
        write_header = not os.path.exists(output_path)
        with open(output_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("config,checkpoint,postprocess,mAP,cmc1,cmc5,cmc10\n")
            f.write(
                f"{result['config']},{result['checkpoint']},{result['postprocess']},"
                f"{result['mAP']:.6f},{result['cmc1']:.6f},{result['cmc5']:.6f},{result['cmc10']:.6f}\n"
            )
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def _load_eval_data(eval_cfg):
    data_loaders = make_data_loader(eval_cfg)
    if len(data_loaders) < 2:
        raise ValueError("make_data_loader check")
    _, val_loader, num_query, num_classes = data_loaders
    return val_loader, num_query, num_classes


def _get_feats(eval_cfg, checkpoint_path, device):
    val_loader, num_query, num_classes = _load_eval_data(eval_cfg)
    model = load_model(eval_cfg, checkpoint_path, device, num_classes)
    feats, pids, camids, img_paths = extract_features(model, val_loader, device)
    if getattr(eval_cfg.TEST, "FEAT_NORM", "no") == "yes":
        feats = l2_normalize(feats)
    return feats, pids, camids, img_paths, num_query


def do_test(
    cfg,
    checkpoint_path,
    postprocess="none",
    output_path: Optional[str] = None,
    orientation_cfg=None,
    orientation_ckpt=None,
    camera_cfg=None,
    camera_ckpt=None,
):
    logger = setup_logger("reid_baseline.test", cfg.OUTPUT_DIR, is_train=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feats, pids, camids, img_paths, num_query = _get_feats(cfg, checkpoint_path, device)

    q_feats = feats[:num_query]
    q_pids = pids[:num_query]
    q_camids = camids[:num_query]

    g_feats = feats[num_query:]
    g_pids = pids[num_query:]
    g_camids = camids[num_query:]
    g_img_paths = img_paths[num_query:]

    if postprocess == "rerank":
        distmat = rerank_distmat(q_feats, g_feats, device)
    else:
        distmat = cosine_distmat(q_feats, g_feats)

    if orientation_cfg and orientation_ckpt:
        o_feats, _, _, _, o_num_query = _get_feats(orientation_cfg, orientation_ckpt, device)
        o_q = o_feats[:o_num_query]
        o_g = o_feats[o_num_query:]
        dist_o = cosine_distmat(o_q, o_g)
        distmat = distmat - cfg.TEST.ORI_LAMBDA * dist_o

    if camera_cfg and camera_ckpt:
        c_feats, _, _, _, c_num_query = _get_feats(camera_cfg, camera_ckpt, device)
        c_q = c_feats[:c_num_query]
        c_g = c_feats[c_num_query:]
        dist_c = cosine_distmat(c_q, c_g)
        distmat = distmat - cfg.TEST.CAM_LAMBDA * dist_c

    if getattr(cfg.TEST, "TRACKLET_RERANK", False):
        tracks = load_tracklets(getattr(cfg.TEST, "TRACKS_PATH", ""))
        distmat = apply_tracklet_min_dist(distmat, g_img_paths, tracks)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC Rank-{r}: {cmc[r-1]:.1%}")

    result = {
        "config": getattr(cfg, "CONFIG_NAME", "unknown"),
        "checkpoint": checkpoint_path,
        "postprocess": postprocess,
        "mAP": float(mAP),
        "cmc1": float(cmc[0]),
        "cmc5": float(cmc[4]) if len(cmc) > 4 else float("nan"),
        "cmc10": float(cmc[9]) if len(cmc) > 9 else float("nan"),
    }
    _save_result(output_path, result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Evaluation")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint", type=str)
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=["none", "rerank"],
        help="post-processing method",
    )
    parser.add_argument("--orientation_config", default="", help="orientation model config", type=str)
    parser.add_argument("--orientation_checkpoint", default="", help="orientation checkpoint", type=str)
    parser.add_argument("--camera_config", default="", help="camera model config", type=str)
    parser.add_argument("--camera_checkpoint", default="", help="camera checkpoint", type=str)
    parser.add_argument(
        "--output",
        default="",
        help="save results to .json/.jsonl/.csv (optional)",
        type=str,
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
        cfg.CONFIG_NAME = os.path.basename(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    output_path = args.output if args.output else None
    orientation_cfg = None
    camera_cfg = None
    if args.orientation_config:
        orientation_cfg = cfg.clone()
        orientation_cfg.merge_from_file(args.orientation_config)
        orientation_cfg.CONFIG_NAME = os.path.basename(args.orientation_config)
    if args.camera_config:
        camera_cfg = cfg.clone()
        camera_cfg.merge_from_file(args.camera_config)
        camera_cfg.CONFIG_NAME = os.path.basename(args.camera_config)

    do_test(
        cfg,
        args.checkpoint,
        postprocess=args.postprocess,
        output_path=output_path,
        orientation_cfg=orientation_cfg,
        orientation_ckpt=args.orientation_checkpoint or None,
        camera_cfg=camera_cfg,
        camera_ckpt=args.camera_checkpoint or None,
    )

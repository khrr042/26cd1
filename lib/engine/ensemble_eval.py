import argparse
import json
import os

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


def main():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")
    parser.add_argument("--spec", required=True, help="path to ensemble spec json", type=str)
    parser.add_argument("--output", default="", help="save results to .json/.jsonl/.csv", type=str)
    args = parser.parse_args()

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    models = spec.get("models", [])
    if not models:
        raise ValueError("spec.models is empty")

    postprocess = spec.get("postprocess", "none")
    tracklets_path = spec.get("tracklets_path", "")
    fusion = spec.get("fusion", {})
    ori_lambda = fusion.get("ori_lambda", 0.2)
    cam_lambda = fusion.get("cam_lambda", 0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    distmats = []
    pids = camids = img_paths = None
    num_query = None

    for entry in models:
        cfg_path = entry["config"]
        ckpt_path = entry["checkpoint"]
        eval_cfg = cfg.clone()
        eval_cfg.merge_from_file(cfg_path)
        eval_cfg.freeze()

        feats, pids, camids, img_paths, num_query = _get_feats(eval_cfg, ckpt_path, device)
        q_feats = feats[:num_query]
        g_feats = feats[num_query:]

        if postprocess == "rerank":
            distmat = rerank_distmat(q_feats, g_feats, device)
        else:
            distmat = cosine_distmat(q_feats, g_feats)

        distmats.append(distmat)

    distmat = np.mean(distmats, axis=0)

    if "orientation" in spec and spec["orientation"].get("checkpoint"):
        o_cfg = cfg.clone()
        o_cfg.merge_from_file(spec["orientation"]["config"])
        o_cfg.freeze()
        o_feats, _, _, _, o_num_query = _get_feats(o_cfg, spec["orientation"]["checkpoint"], device)
        o_q = o_feats[:o_num_query]
        o_g = o_feats[o_num_query:]
        distmat = distmat - ori_lambda * cosine_distmat(o_q, o_g)

    if "camera" in spec and spec["camera"].get("checkpoint"):
        c_cfg = cfg.clone()
        c_cfg.merge_from_file(spec["camera"]["config"])
        c_cfg.freeze()
        c_feats, _, _, _, c_num_query = _get_feats(c_cfg, spec["camera"]["checkpoint"], device)
        c_q = c_feats[:c_num_query]
        c_g = c_feats[c_num_query:]
        distmat = distmat - cam_lambda * cosine_distmat(c_q, c_g)

    if tracklets_path:
        tracks = load_tracklets(tracklets_path)
        distmat = apply_tracklet_min_dist(distmat, img_paths[num_query:], tracks)

    cmc, mAP = eval_func(distmat, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])
    logger = setup_logger("reid_ensemble.test", None, is_train=False)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC Rank-{r}: {cmc[r-1]:.1%}")

    result = {
        "spec": os.path.basename(args.spec),
        "postprocess": postprocess,
        "mAP": float(mAP),
        "cmc1": float(cmc[0]),
        "cmc5": float(cmc[4]) if len(cmc) > 4 else float("nan"),
        "cmc10": float(cmc[9]) if len(cmc) > 9 else float("nan"),
    }

    if args.output:
        if args.output.lower().endswith(".jsonl"):
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        elif args.output.lower().endswith(".csv"):
            write_header = not os.path.exists(args.output)
            with open(args.output, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("spec,postprocess,mAP,cmc1,cmc5,cmc10\n")
                f.write(
                    f"{result['spec']},{result['postprocess']},"
                    f"{result['mAP']:.6f},{result['cmc1']:.6f},{result['cmc5']:.6f},{result['cmc10']:.6f}\n"
                )
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

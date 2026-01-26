import torch
import numpy as np
from lib.config import cfg
from lib.data import make_data_loader
from lib.models import build_model
from lib.utils.metrics import eval_func
from lib.utils.utils import setup_logger

def do_test(cfg, checkpoint_path):
    logger = setup_logger("reid_baseline.test", cfg.OUTPUT_DIR, is_train=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loaders = make_data_loader(cfg)
    if len(data_loaders) < 2:
        raise ValueError("make_data_loader check")
    _, val_loader, num_query, num_classes = data_loaders

    model = build_model(cfg, num_classes)
    if model is None:
        raise ValueError("build model returned None")
    model = model.to(device)
    model.eval()

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    feats, pids, camids = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) < 3:
                raise ValueError("DataLoader batch is smaller than pred")
            img, pid, camid = batch[:3]
            if img is None:
                raise ValueError("DataLoader img none")
            img = img.to(device)
            feat = model(img)
            if isinstance(feat, tuple):
                feat = feat[0]
            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))

    feats = torch.cat(feats, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)

    q_feats = feats[:num_query]
    q_pids = pids[:num_query]
    q_camids = camids[:num_query]

    g_feats = feats[num_query:]
    g_pids = pids[num_query:]
    g_camids = camids[num_query:]

    distmat = 1 - torch.mm(q_feats, g_feats.t())

    cmc, mAP = eval_func(distmat.numpy(), q_pids, g_pids, q_camids, g_camids)
    
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC Rank-{r}: {cmc[r-1]:.1%}")

if __name__ == "__main__":
    checkpoint_path = "path/to/checkpoint.pth"
    do_test(cfg, checkpoint_path)

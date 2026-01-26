import torch
import numpy as np

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    if torch.is_tensor(distmat):
        distmat = distmat.cpu().numpy()
    if torch.is_tensor(q_pids):
        q_pids = q_pids.cpu().numpy()
    if torch.is_tensor(g_pids):
        g_pids = g_pids.cpu().numpy()
    if torch.is_tensor(q_camids):
        q_camids = q_camids.cpu().numpy()
    if torch.is_tensor(g_camids):
        g_camids = g_camids.cpu().numpy()
        
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: max_rank is larger than gallery size, reduce to {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.) for i, x in enumerate(tmp_cmc)])
        tmp_cmc = tmp_cmc * raw_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

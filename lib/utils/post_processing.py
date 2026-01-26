# encoding: utf-8
import torch
import torch.nn.functional as F
import numpy as np
import os

def comput_distmat(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = (qf ** 2).sum(dim=1, keepdim=True) + (gf ** 2).sum(dim=1) - 2 * qf @ gf.t()
    return distmat

def database_aug(gf, top_k=10):
    distmat = comput_distmat(gf, gf)
    indices = torch.argsort(distmat, dim=1)
    expanded_gf = torch.stack([gf[indices[i, :top_k]].mean(dim=0) for i in range(gf.size(0))])
    return expanded_gf

def average_query_expansion(qf, feats, top_k=6):
    distmat = comput_distmat(qf, feats)
    indices = torch.argsort(distmat, dim=1)
    expanded_qf = torch.stack([feats[indices[i, :top_k]].mean(dim=0) for i in range(qf.size(0))])
    return expanded_qf

def alpha_query_expansion(qf, feats, alpha=3.0, top_k=10):
    distmat = comput_distmat(qf, feats)
    indices = torch.argsort(distmat, dim=1)
    expanded_qf = torch.stack([feats[indices[i, :top_k]].mean(dim=0) for i in range(qf.size(0))])
    return expanded_qf

def track_aug(feats, tracks, img_paths):
    assert len(feats) == len(img_paths), 'len(feats) != len(img_paths)'

    lookup = {os.path.basename(name): i for i, track in enumerate(tracks) for name in track}
    img_names = [os.path.basename(p) for p in img_paths]

    average_seq = [[] for _ in range(len(tracks))]
    for idx, name in enumerate(img_names):
        if name in lookup:
            average_seq[lookup[name]].append(idx)

    for seq in average_seq:
        if len(seq) == 0: continue
        track_feats = feats[seq, :]
        dist = comput_distmat(track_feats, track_feats)
        weights = 1 / (dist + 1e-6)
        feats[seq, :] = (weights @ track_feats) / weights.sum(dim=1, keepdim=True)
    
    return F.normalize(feats, dim=1, p=2)

def pca_whiten(qf, gf, dim=256):
    X = torch.cat([gf, qf], dim=0)
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    components = Vh[:dim]
    gf_new = (gf - X_mean) @ components.T
    qf_new = (qf - X_mean) @ components.T
    return F.normalize(qf_new, dim=1), F.normalize(gf_new, dim=1)

def generate_track_idxs(gallery_names, tracks):
    img_to_idx = {name: i for i, name in enumerate(gallery_names)}
    track_idxs = [[img_to_idx[name] for name in track if name in img_to_idx] for track in tracks]
    return track_idxs

def generate_track_distmat(distmat, track_idxs):
    track_distmat = [distmat[:, idx].min(1, keepdim=True)[0] for idx in track_idxs if len(idx) > 0]
    return torch.cat(track_distmat, dim=1)

def rerank_gpu(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea], dim=0)
    
    distmat = comput_distmat(feat, feat)
    original_dist = distmat / distmat.max()
    del feat, distmat

    original_dist = original_dist.cpu().numpy()
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward = initial_rank[i, :k1+1]
        backward = initial_rank[forward, :k1+1]
        fi = np.where(backward == i)[1]
        k_reciprocal = forward[fi]
        k_reciprocal_exp = k_reciprocal.copy()
        for candidate in k_reciprocal:
            candidate_forward = initial_rank[candidate, :int(np.around(k1/2))+1]
            candidate_backward = initial_rank[candidate_forward, :int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward == candidate)[1]
            candidate_k_recip = candidate_forward[fi_candidate]
            if len(np.intersect1d(candidate_k_recip, k_reciprocal)) > 2/3*len(candidate_k_recip):
                k_reciprocal_exp = np.append(k_reciprocal_exp, candidate_k_recip)
        k_reciprocal_exp = np.unique(k_reciprocal_exp)
        weight = np.exp(-original_dist[i, k_reciprocal_exp])
        V[i, k_reciprocal_exp] = weight / weight.sum()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = V[initial_rank[i, :k2], :].mean(axis=0)
        V = V_qe

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(galFea.size(0)+query_num)]
    jaccard_dist = np.zeros((query_num, galFea.size(0)), dtype=np.float32)
    for i in range(query_num):
        indNonZero = np.where(V[i, :] != 0)[0]
        temp_min = np.zeros((1, galFea.size(0)), dtype=np.float32)
        indImages = [invIndex[ind] for ind in indNonZero]
        for j, inds in enumerate(indImages):
            temp_min[0, inds] += np.minimum(V[i, indNonZero[j]], V[inds, indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, query_num:] * lambda_value
    return final_dist

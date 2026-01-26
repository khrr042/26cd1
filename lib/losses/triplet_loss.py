import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def hard_example_mining(dist_mat, labels):
    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    return dist_ap, dist_an

class TripletLoss(object):
    def __init__(self, margin=0.3):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = F.normalize(global_feat, dim=1, p=2)
            
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from .datasets import init_dataset
from .datasets.bases import ImageDataset
from .transforms import build_transforms
from .sampler import RandomIdentitySampler

def train_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = dataset.train
    val_set = dataset.query + dataset.gallery

    train_dataset = ImageDataset(train_set, transform=train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        print(f"Using RandomIdentitySampler: {cfg.DATALOADER.NUM_INSTANCE} instances per ID")
        train_sampler = RandomIdentitySampler(
            data_source=train_set,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_instances=cfg.DATALOADER.NUM_INSTANCE
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=num_workers, collate_fn=train_collate_fn,
        pin_memory=True, drop_last=True
    )

    val_dataset = ImageDataset(val_set, transform=val_transforms)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, len(dataset.query), num_classes
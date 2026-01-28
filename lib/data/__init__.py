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

def _ensure_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_dataset(name, root, validation_split_id=None):
    kwargs = {}
    if name == "cityflow" and validation_split_id is not None:
        kwargs["validation_split_id"] = validation_split_id
    return init_dataset(name, root=root, **kwargs)


def _remap_labels(train_set, label_map):
    remapped = []
    for img_path, label, camid in train_set:
        remapped.append((img_path, label_map[label], camid))
    return remapped


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_names = getattr(cfg.DATASETS, "TRAIN_NAMES", cfg.DATASETS.NAMES)
    train_names = _ensure_list(train_names)
    test_name = getattr(cfg.DATASETS, "TEST_NAME", train_names[0])
    label_type = getattr(cfg.DATASETS, "LABEL_TYPE", "pid")
    val_split = getattr(cfg.DATASETS, "VALIDATION_SPLIT_ID", None)

    train_datasets = [
        _build_dataset(name, cfg.DATASETS.ROOT_DIR, validation_split_id=val_split)
        for name in train_names
    ]
    test_dataset = _build_dataset(test_name, cfg.DATASETS.ROOT_DIR, validation_split_id=val_split)

    train_set = []
    if label_type == "pid":
        pid_offset = 0
        for ds in train_datasets:
            for img_path, pid, camid in ds.train:
                train_set.append((img_path, pid + pid_offset, camid))
            pid_offset += getattr(ds, "num_train_pids", 0)
        num_classes = pid_offset
    else:
        raw_labels = []
        for ds in train_datasets:
            for img_path, pid, camid in ds.train:
                if label_type == "camid":
                    raw_labels.append(camid)
                    train_set.append((img_path, camid, camid))
                elif label_type == "orientation":
                    meta = getattr(ds, "meta", {})
                    orientation = meta.get(img_path, {}).get("orientation", -1)
                    if orientation < 0:
                        raise ValueError(f"Missing orientation label for {img_path}")
                    raw_labels.append(orientation)
                    train_set.append((img_path, orientation, camid))
                else:
                    raise ValueError(f"Unsupported label type: {label_type}")

        label_map = {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}
        train_set = _remap_labels(train_set, label_map)
        num_classes = len(label_map)

    val_set = test_dataset.query + test_dataset.gallery

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

    return train_loader, val_loader, len(test_dataset.query), num_classes

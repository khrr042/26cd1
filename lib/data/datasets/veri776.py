import os
import os.path as osp
import re

from .bases import BaseImageDataset


class VeRi776(BaseImageDataset):
    dataset_dir = "VeRi-776"

    def __init__(self, root="./data", verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "image_train")
        self.query_dir = osp.join(self.dataset_dir, "image_query")
        self.gallery_dir = osp.join(self.dataset_dir, "image_test")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> VeRi-776 Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        if not osp.isdir(dir_path):
            raise RuntimeError(f"{dir_path} not found.")

        img_paths = sorted(
            [p for p in os.listdir(dir_path) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        pattern = re.compile(r"([-\d]+)_c(\d+)")

        dataset = []
        pid_container = set()
        for img_name in img_paths:
            match = pattern.search(img_name)
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            pid_container.add(pid)

        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        for img_name in img_paths:
            match = pattern.search(img_name)
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            img_path = osp.join(dir_path, img_name)
            dataset.append((img_path, pid, camid))

        return dataset

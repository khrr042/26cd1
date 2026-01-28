import os
import os.path as osp
import re
import xml.etree.ElementTree as ET

from .bases import BaseImageDataset


class VehicleX(BaseImageDataset):
    dataset_dir = "VehicleX"

    def __init__(self, root="./data", verbose=True, list_path=None, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "sys_image_train")
        self.train_label_path = osp.join(self.dataset_dir, "train_label.xml")
        self.list_path = list_path or osp.join(self.dataset_dir, "train_list.txt")

        if osp.isfile(self.train_label_path):
            train = self._process_dir_xml(self.train_dir, self.train_label_path)
        elif osp.isfile(self.list_path):
            train = self._process_list(self.list_path)
        else:
            train = self._process_dir(self.train_dir)

        self.train = self._relabel(train)
        self.query = []
        self.gallery = []

        if verbose:
            print("=> VehicleX Loaded (train only)")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _process_dir_xml(self, img_dir, label_path):
        if not osp.exists(label_path):
            raise RuntimeError(f"{label_path} not found.")

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            xml_content = re.sub(r'encoding="[^"]+"', "", f.read())
        root = ET.fromstring(xml_content)

        items = root.find("Items")
        if items is None:
            items = root

        dataset = []
        for obj in items:
            image_name = obj.attrib.get("imageName")
            if image_name is None:
                continue
            img_path = osp.join(img_dir, image_name)
            pid = int(obj.attrib.get("vehicleID", -1))

            cam_raw = obj.attrib.get("cameraID", "0")
            m = re.findall(r"\d+", cam_raw)
            camid = int(m[0]) if m else 0

            if pid == -1:
                continue
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_list(self, list_path):
        dataset = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                rel_path = parts[0]
                pid = int(parts[1])
                camid = int(parts[2]) if len(parts) > 2 else 0
                img_path = rel_path
                if not osp.isabs(img_path):
                    img_path = osp.join(self.dataset_dir, rel_path)
                dataset.append((img_path, pid, camid))
        return dataset

    def _process_dir(self, dir_path):
        if not osp.isdir(dir_path):
            raise RuntimeError(f"{dir_path} not found.")
        img_paths = sorted(
            [p for p in os.listdir(dir_path) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        pattern = re.compile(r"([-\d]+)_c(\d+)")
        dataset = []
        for img_name in img_paths:
            match = pattern.search(img_name)
            if not match:
                continue
            pid, camid = map(int, match.groups())
            if pid == -1:
                continue
            img_path = osp.join(dir_path, img_name)
            dataset.append((img_path, pid, camid))
        return dataset

    def _relabel(self, dataset):
        pids = set()
        for _, pid, _ in dataset:
            pids.add(pid)
        pid_dict = {pid: i for i, pid in enumerate(sorted(pids))}
        new_dataset = []
        for img_path, pid, camid in dataset:
            new_dataset.append((img_path, pid_dict[pid], camid))
        return new_dataset


class VehicleXTranslated(VehicleX):
    dataset_dir = "VehicleX_Translated"

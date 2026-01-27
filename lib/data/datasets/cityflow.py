import os.path as osp
import re
import random
import copy
import xml.etree.ElementTree as ET
from .bases import BaseImageDataset

class CityFlow(BaseImageDataset):
    dataset_dir = 'AIC21_Track2_ReID'

    def __init__(self, root='./data', verbose=True, validation_split_id=10, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.train_label_path = osp.join(self.dataset_dir, 'train_label.xml')

        full_train = self._process_dir_xml(self.train_dir, self.train_label_path, is_train=True)
        
        train_list, query_list, gallery_list = self._split_train_val(full_train, num_val_ids=validation_split_id)

        train = self._relabel(train_list)
        
        
        self.train = train
        self.query = query_list
        self.gallery = gallery_list

        if verbose:
            print(f"=> AICity20 Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _split_train_val(self, dataset, num_val_ids=20):
        pids = sorted(list(set([item[1] for item in dataset])))
        random.seed(1234)
        random.shuffle(pids)
        
        val_pids = set(pids[-num_val_ids:])
        train_pids = set(pids[:-num_val_ids])
        
        train_data = []
        val_data_candidates = []
        
        for item in dataset:
            if item[1] in train_pids:
                train_data.append(item)
            else:
                val_data_candidates.append(item)
                
        query_data = []
        gallery_data = []
        
        val_dict = {}
        for img, pid, camid in val_data_candidates:
            key = (pid, camid)
            if key not in val_dict: val_dict[key] = []
            val_dict[key].append(img)
            
        for key, imgs in val_dict.items():
            pid, camid = key
            random.shuffle(imgs)
            query_data.append((imgs[0], pid, camid))
            for img in imgs[1:]:
                gallery_data.append((img, pid, camid))
                
        return train_data, query_data, gallery_data

    def _relabel(self, dataset):
        pids = set()
        for _, pid, _ in dataset:
            pids.add(pid)
        self.pid_dict = {pid: i for i, pid in enumerate(sorted(pids))}
        new_dataset = []
        for img_path, pid, camid in dataset:
            new_pid = self.pid_dict[pid]
            new_dataset.append((img_path, new_pid, camid))
        return new_dataset

    def _process_dir_xml(self, img_dir, label_path, is_train=False):
        if not osp.exists(label_path):
            raise RuntimeError(f"{label_path} not found.")
        
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = re.sub(r'encoding="[^"]+"', '', f.read())
        root = ET.fromstring(xml_content)
        
        items = root.find('Items')
        if items is None: items = root
            
        dataset = []
        for obj in items:
            image_name = obj.attrib['imageName']
            img_path = osp.join(img_dir, image_name)
            pid = int(obj.attrib.get('vehicleID', -1))
            
            cam_raw = obj.attrib.get('cameraID', '0')
            m = re.findall(r'\d+', cam_raw)
            camid = int(m[0]) if m else 0
            
            if pid == -1: continue
            dataset.append((img_path, pid, camid))
        return dataset
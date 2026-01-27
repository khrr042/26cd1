import os.path as osp
import re
import xml.etree.ElementTree as ET
from .bases import BaseImageDataset

class CityFlow(BaseImageDataset):
    dataset_dir = 'AIC21_Track2_ReID'

    def __init__(self, root='./data', verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self.train_label_path = osp.join(self.dataset_dir, 'train_label.xml')
        self.query_label_path = osp.join(self.dataset_dir, 'query_label.xml')
        self.gallery_label_path = osp.join(self.dataset_dir, 'test_label.xml')

        self._check_before_run()

        train = self._process_dir_xml(self.train_dir, self.train_label_path, is_train=True)
        query = self._process_dir_xml(self.query_dir, self.query_label_path, is_train=False)
        gallery = self._process_dir_xml(self.gallery_dir, self.gallery_label_path, is_train=False)

        self.pid_dict = {}
        train = self._relabel(train)

        if verbose:
            print(f"=> AIC21 Track2 Loaded: {self.dataset_dir}")
            self.print_dataset_statistics(train, query, gallery)

        self.train, self.query, self.gallery = train, query, gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        required = [
            self.dataset_dir,
            self.train_dir, self.query_dir, self.gallery_dir,
            self.train_label_path, self.query_label_path, self.gallery_label_path,
        ]
        for p in required:
            if not osp.exists(p):
                raise RuntimeError(f"Missing: {p}")
    
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
        dataset = []
        try:
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                xml_content = f.read()
            xml_content = re.sub(r'encoding="[^"]+"', '', xml_content)
            root = ET.fromstring(xml_content)
        except Exception as e:
            print(f"Error processing XML file {label_path}: {e}")
            raise e

        items = root.find('Items')
        if items is None:
            if root.tag == 'Items':
                items = root
            else:
                for child in root:
                    if child.tag.lower() == 'items':
                        items = child
                        break
        
        if items is None:
             raise RuntimeError(f"Invalid XML format (Items tag not found): {label_path}")

        for obj in items:
            image_name = obj.attrib['imageName']
            img_path = osp.join(img_dir, image_name)
            
            if not osp.exists(img_path):
                continue 

            pid_str = obj.attrib.get('vehicleID')
            if pid_str:
                pid = int(pid_str)
            else:
                if is_train:
                    continue
                pid = -1

            cam_raw = obj.attrib.get('cameraID', '0')
            m = re.findall(r'\d+', cam_raw)
            camid = int(m[0]) if m else 0

            dataset.append((img_path, pid, camid))
            
        return dataset
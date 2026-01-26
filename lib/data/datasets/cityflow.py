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

        train = self._process_dir_xml(self.train_dir, self.train_label_path, relabel=True)
        query = self._process_dir_xml(self.query_dir, self.query_label_path, relabel=False)
        gallery = self._process_dir_xml(self.gallery_dir, self.gallery_label_path, relabel=False)

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

    def _process_dir_xml(self, img_dir, label_path, relabel=False):
        dataset = []
        tree = ET.parse(label_path)
        items = tree.find('Items')
        if items is None:
            raise RuntimeError(f"Invalid XML format (Items not found): {label_path}")

        for obj in items:
            image_name = obj.attrib['imageName']
            img_path = osp.join(img_dir, image_name)
            if not osp.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            pid = int(obj.attrib['vehicleID'])

            cam_raw = obj.attrib.get('cameraID', '0')
            m = re.findall(r'\d+', cam_raw)
            camid = int(m[0]) if m else 0

            dataset.append((img_path, pid, camid))
        return dataset

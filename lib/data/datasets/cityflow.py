import os.path as osp
import glob
import re
from .bases import BaseImageDataset

class CityFlow(BaseImageDataset):
    """
    CityFlowV2 Dataset
    
    Dataset structure:
    data/
        CityFlowV2/
            image_train/
            image_query/
            image_test/
            name_train.txt
            name_query.txt
            name_test.txt
    """
    dataset_dir = 'CityFlowV2'

    def __init__(self, root='./data', verbose=True, **kwargs):
        super(CityFlow, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, is_train=True)
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)

        if verbose:
            print("=> CityFlowV2 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'(\d+)_c(\d+)')

        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            
            match = pattern.search(img_name)
            
            if match:
                pid, camid = map(int, match.groups())
                
                if is_train:
                    pid -= 1
                camid -= 1

                data.append((img_path, pid, camid))
            else:
                print(f"Warning: Skipping file with unexpected format: {img_name}")
                continue

        return data

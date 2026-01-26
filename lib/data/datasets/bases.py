import numpy as np
from PIL import Image, ImageFile
import torch.utils.data as data
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'.")
            pass
    return img

class BaseImageDataset(object):
    def get_imagedata_info(self, data):
        pids, cams = [], []
        for item in data:
            pid, camid = item[1], item[2]
            pids.append(pid)
            cams.append(camid)
        return len(set(pids)), len(data), len(set(cams)), 0

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, _ = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

class ImageDataset(data.Dataset):
    """Generic Image Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
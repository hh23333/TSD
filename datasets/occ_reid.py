import glob
import re
import urllib
import zipfile

import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset

class Occluded_REID(BaseImageDataset):
    # dataset_dir = 'Occluded_Duke'
    dataset_dir = 'ICME2018_Occluded-Person-Reidentification_datasets/Occluded_REID'
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Occluded_REID, self).__init__()
        # self.dataset_dir = self.root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir=osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir=osp.join(self.dataset_dir, 'whole_body_images')
        self.pid_begin = pid_begin
        train = []
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path,'*','*.jpg'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 1))
        return data
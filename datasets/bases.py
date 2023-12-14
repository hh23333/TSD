from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path
        # return img, pid, camid, trackid,img_path.split('/')[-1]
    
import cv2
import numpy as np
    
def read_image_cv2(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img

def read_masks(masks_path):
    """Reads part-based masks information from path.

    Args:
        path (str): path to an image. Part-based masks information is stored in a .npy file with image name as prefix

    Returns:
        Numpy array of size N x H x W where N is the number of part-based masks
    """

    got_masks = False
    if not osp.exists(masks_path):
        raise IOError('Masks file"{}" does not exist'.format(masks_path))
    while not got_masks:
        try:
            masks = np.load(masks_path)
            masks = np.transpose(masks, (1, 2, 0))
            got_masks = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(masks_path)
            )
    return masks

class ImageDataset_M(Dataset):
    def __init__(self, dataset, transform=None, with_mask=True):
        self.dataset = dataset
        self.transform = transform
        self.with_mask = with_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image_cv2(img_path)
        
        load_dict = {'image': img}
        #
        if self.with_mask:
            mask_dir = osp.join(osp.dirname(osp.dirname(img_path)), 'masks/pifpaf_maskrcnn_filtering')
            mask_path = osp.join(img_path.split('/')[-2], img_path.split('/')[-1]+'.confidence_fields.npy')
            # mask_path = osp.join(img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]+'.npy')
            mask_path = osp.join(mask_dir, mask_path)
            mask = read_masks(mask_path)
            load_dict.update({'mask': mask})

        if self.transform is not None:
            result = self.transform(**load_dict)
        
        if self.with_mask:
            return result['image'], pid, camid, trackid, img_path, result['mask']
        else:
            return result['image'], pid, camid, trackid, img_path

import torch
import torch.utils.data as Data

import os
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import rotate
import os.path as osp
import sys
import random
import numpy as np
import cv2


class TrainDataset(Data.Dataset):
    def __init__(self, args, mode='train', img_norm_cfg=None):
        self.args = args
        dataset_dir = args.dataset_root + '/' + args.dataset

        if args.dataset in ['Maritime_sirst', 'dataset']:
            if mode == 'train':
                self.imgs_dir = osp.join(dataset_dir, 'trainval', 'images')
                self.label_dir = osp.join(dataset_dir, 'trainval', 'masks')
            else:
                self.imgs_dir = osp.join(dataset_dir, 'detect', 'images')
                self.label_dir = osp.join(dataset_dir, 'detect', 'masks')

            self.names = []
            for filename in os.listdir(self.imgs_dir):
                if filename.endswith('png'):
                    base_name, _ = osp.splitext(filename)
                    self.names.append(base_name)
        else:
            if mode == 'train':
                txtfile = 'trainval.txt'
            elif mode == 'val' or mode == 'vis':
                txtfile = 'test.txt'

            self.list_dir = osp.join(dataset_dir, txtfile)
            self.imgs_dir = osp.join(dataset_dir, 'images')
            self.label_dir = osp.join(dataset_dir, 'masks')

            self.names = []
            with open(self.list_dir, 'r') as f:
                self.names += [line.strip() for line in f.readlines()]

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(args.dataset, dataset_dir)

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        if self.args.dataset == 'NUAA-SIRST':
            label_path = osp.join(self.label_dir, name + '_pixels0.png')
        else:
            label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask = np.array(mask, dtype=np.float32) / 255.0

            img, mask = self._random_crop(np.array(img), np.array(mask), self.crop_size, pos_prob=0.5)
            img, mask = self._sync_transform(img, mask)

            img_batch, mask_batch = img[np.newaxis, :], mask[np.newaxis, :]
            img_batch = torch.from_numpy(np.ascontiguousarray(img_batch)).to(torch.float32)
            mask_batch = torch.from_numpy(np.ascontiguousarray(mask_batch)).to(torch.float32)
            return img_batch, mask_batch

        elif self.mode == 'val':
            img = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask = np.array(mask, dtype=np.float32) / 255.0
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]

            img = PadImg(img)
            mask = PadImg(mask)

            img, mask = img[np.newaxis, :], mask[np.newaxis, :]

            img = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32)
            if img.shape != mask.shape:
                print('img!=mask in dataset')
            return img, mask

        else:
            raise ValueError("Unkown self.mode")

    def __len__(self):
        return len(self.names)

    def _random_crop(self, img, mask, patch_size, pos_prob=None):
        h, w = img.shape
        if min(h, w) < patch_size:
            img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                         mode='constant')
            mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                          mode='constant')
            h, w = img.shape

        while 1:
            h_start = random.randint(0, h - patch_size)
            h_end = h_start + patch_size
            w_start = random.randint(0, w - patch_size)
            w_end = w_start + patch_size

            img_patch = img[h_start:h_end, w_start:w_end]
            mask_patch = mask[h_start:h_end, w_start:w_end]

            if pos_prob == None or random.random() > pos_prob:
                break
            elif mask_patch.sum() > 0:
                break

        return img_patch, mask_patch

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:  # 水平反转
            img = img[::-1, :]
            mask = mask[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            img = img[:, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            img = img.transpose(1, 0)
            mask = mask.transpose(1, 0)
        # random rotate
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = rotate(img, angle, reshape=False, order=1)  # bilinear
            mask = rotate(mask, angle, reshape=False, order=0)  # nearest

        return img, mask


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1k':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'SIRST2':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'NUDT-SIRST-Sea-Light':
        img_norm_cfg = dict(mean=21.39715576171875, std=10.919337272644043)
    elif dataset_name == 'Maritime_sirst':
        img_norm_cfg = dict(mean=36.6230583190918, std=13.484057426452637)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST5':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST6':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST7':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + '/trainval' + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + '/test' + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg

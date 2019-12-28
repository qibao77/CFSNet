import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import codes.data.util as util


class LRGTDataset(data.Dataset):

    def __init__(self, opt):
        super(LRGTDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_GT = None
        self.LR_env = None  # environment for lmdb
        self.GT_env = None

        # read image list
        self.GT_env, self.paths_GT = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LR and self.paths_GT:
            assert len(self.paths_LR) == len(self.paths_GT), \
                'GT and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_GT))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        patch_size = self.opt['patch_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(self.GT_env, GT_path)

        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < patch_size or W < patch_size:
                img_GT = cv2.resize(
                    np.copy(img_GT), (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)
                img_LR = cv2.resize(
                    np.copy(img_LR), (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                if img_LR.ndim == 2: img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            # randomly crop
            rnd_h = random.randint(0, max(0, H - patch_size))
            rnd_w = random.randint(0, max(0, W - patch_size))
            img_LR = img_LR[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
            img_GT = img_GT[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment([img_LR, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path
        return {'LR': img_LR, 'GT': img_GT, 'LR_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)

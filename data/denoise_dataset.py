import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import codes.data.util as util

class GTDataset(data.Dataset):

    def __init__(self, opt):
        super(GTDataset, self).__init__()
        self.opt = opt
        self.paths_GT = None
        self.GT_env = None

        # read image list
        self.GT_env, self.paths_GT = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

    def __getitem__(self, index):
        patch_size = self.opt['patch_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(self.GT_env, GT_path)

        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        if self.opt['phase'] == 'train':
            #image augmentation : scale
            if self.opt['use_scale']:
                H, W, _ = img_GT.shape
                scale = random.randint(8,11)/10.0
                img_GT = cv2.resize(np.copy(img_GT), (int(H * scale), int(W * scale)), interpolation=cv2.INTER_CUBIC)
                if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)

            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < patch_size or W < patch_size:
                img_GT = cv2.resize(np.copy(img_GT), (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
                if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)
            H, W, C = img_GT.shape

            # randomly crop
            rnd_h = random.randint(0, max(0, H - patch_size))
            rnd_w = random.randint(0, max(0, W - patch_size))
            img_GT = img_GT[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

            # augmentation : flip, rotate, scale
            img_GT = util.augment([img_GT], self.opt['use_flip'], self.opt['use_rot'])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        return {'GT': img_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)

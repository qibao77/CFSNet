import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch
import numpy as np

from settings import options as option
from utils import util
from data.util import bgr2ycbcr
from utils.logger import Logger, PrintLogger
from data import create_dataloader,create_dataset
from models import create_model


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt) #load settings and initialize settings

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
        not key == 'saved_model'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # Redirect all writes to the "txt" file
    sys.stdout = PrintLogger(opt['path']['log'])

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # Create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter {:d}.'.format(current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                print('---------- validation -------------')
                start_time = time.time()

                avg_psnr = 0.0
                avg_ssim =0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['GT_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    out_img = util.tensor2img(visuals['Output'])
                    gt_img = util.tensor2img(visuals['ground_truth'])  # uint8

                    # Save output images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(out_img, save_img_path)

                    # calculate PSNR
                    if len(gt_img.shape) == 2:
                        gt_img = np.expand_dims(gt_img, axis=2)
                        out_img = np.expand_dims(out_img, axis=2)
                    crop_border = opt['scale']
                    cropped_out_img = out_img[crop_border:-crop_border, crop_border:-crop_border, :]
                    cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
                    if gt_img.shape[2] == 3:  # RGB image
                        cropped_out_img_y = bgr2ycbcr(cropped_out_img, only_y=True)
                        cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)
                        avg_psnr += util.psnr(cropped_out_img_y, cropped_gt_img_y)
                        avg_ssim += util.ssim(cropped_out_img_y, cropped_gt_img_y, multichannel=False)
                    else:
                        avg_psnr += util.psnr(cropped_out_img, cropped_gt_img)
                        avg_ssim += util.ssim(cropped_out_img, cropped_gt_img, multichannel=True)

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                print_rlt['psnr'] = avg_psnr
                print_rlt['ssim'] = avg_ssim
                logger.print_format_results('val', print_rlt)
                print('-----------------------------------')

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')

if __name__ == '__main__':
    main()

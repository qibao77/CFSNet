import os
import sys
import time
import argparse
from collections import OrderedDict
import numpy as np
from codes.settings import options as option
from codes.utils import util
from codes.utils.logger import Logger, PrintLogger
from codes.data import create_dataloader,create_dataset
from codes.models import create_model
from codes.data.util import bgr2ycbcr

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
opt = option.parse(parser.parse_args().opt)#load settings and initialize settings
util.mkdirs((path for key, path in opt['path'].items() if not key == 'saved_model'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    need_ground_truth = True

    for data in test_loader:
        need_ground_truth = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        model.feed_data(data, need_ground_truth=need_ground_truth)
        img_path = data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        visuals = model.get_current_visuals(need_ground_truth=need_ground_truth)
        out_img = util.tensor2img(visuals['Output'])

        if need_ground_truth:  # load GT image and calculate psnr
            gt_img = util.tensor2img(visuals['ground_truth'])  # uint8
            if len(gt_img.shape)==2:
                gt_img = np.expand_dims(gt_img, axis=2)
                out_img = np.expand_dims(out_img, axis=2)

            crop_border = test_loader.dataset.opt['scale']
            cropped_out_img = out_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
            psnr = util.psnr(cropped_out_img, cropped_gt_img)
            ssim = util.ssim(cropped_out_img, cropped_gt_img, multichannel=True)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if gt_img.shape[2] == 3:  # RGB image
                cropped_out_img_y = bgr2ycbcr(cropped_out_img, only_y=True)
                cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)
                psnr_y = util.psnr(cropped_out_img_y, cropped_gt_img_y)
                ssim_y = util.ssim(cropped_out_img_y, cropped_gt_img_y, multichannel=False)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}; PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}.'\
                    .format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}.'.format(img_name, psnr, ssim))
        else:
            print(img_name)

        save_img_path = os.path.join(dataset_dir, img_name + '.png')
        util.save_img(out_img, save_img_path)

    if need_ground_truth:
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        path_file = opt['path']['results_root']+ "/" + test_set_name + ".txt"
        print('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.4f} dB; SSIM: {:.4f}\n'\
                .format(test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}\n'\
                .format(ave_psnr_y, ave_ssim_y))

# CFSNet
pytorch code of "CFSNet: Toward a Controllable Feature Space for Image Restoration"（ICCV2019）

#### [arXiv](https://arxiv.org/abs/1904.00634)

<div align=center><img width="360" height="240" src="https://github.com/qibao77/CFSNet/blob/master/figs/coupling_module.png"/></div>

Coupling module architecture.

![](figs/framework.png)

The framework of our proposed controllable feature space network (CFSNet). The details about our proposed CFSNet can be found in [our main paper](https://arxiv.org/abs/1904.00634).

If you find our work useful in your research or publications, please consider citing:

```latex
@article{wang2019cfsnet,
  title={CFSNet: Toward a Controllable Feature Space for Image Restoration},
  author={Wang, Wei and Guo, Ruiming and Tian, Yapeng and Yang, Wenming},
  journal={arXiv preprint arXiv:1904.00634},
  year={2019}
}
```

## Contents
1. [Test](#test)
2. [Results](#results)

## Test

code coming soon...

## Results

#### Visual Results

![](figs/sr_compare.png)

Perceptual and distortion balance of “215”, “211” and “268” (PIRM test dataset) for 4× image super-resolution.

<div align=center><img width="200" height="200" src="https://github.com/qibao77/CFSNet/blob/master/figs/sr_crop.gif"/></div>

Smooth control of distortion and perceptual with control scalar α_in from 0 to 1.

![](figs/color_noise40.png)

Color image denoising results with unknown noise level σ = 15 (first two rows) and σ = 40 (last two rows). α_in = 0.5 and α_in = −0.3 correspond to the highest PSNR results, respectively.

<div align=center><img width="200" height="200" src="https://github.com/qibao77/CFSNet/blob/master/figs/denoise_color_flower_crop.gif"/></div>

Smooth control of noise reduction and detail preservation with control scalar α_in from 0 to 1.

![](figs/jpeg_20.png)

JPEG image artifacts removal results of “house” and “ocean” (LIVE1) with unknown quality factor 20. α_in = 0.5 corresponds to the highest PSNR results, and the best visual results are marked with red boxes.

![](figs/blur_BD16.png)

Visual results of single image super-resolution with unseen degradation (the blur kernel is 7 × 7 Gaussian kernel with standard deviation 1.6, the scale factor is 3). α_in = 0.15 corresponds to the highest PSNR results.

<div align=center><img width="200" height="200" src="https://github.com/qibao77/CFSNet/blob/master/figs/deblur_tiger_crop.gif"/></div>

Smooth control of blur removal and detail sharpening with control scalar α_in from 0 to 0.8.



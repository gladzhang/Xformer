# Xformer: Hybrid X-Shaped Transformer for Image Denoising

Jiale Zhang, [Yulun Zhang](http://yulunzhang.com/), [Jinjin Gu](https://www.jasongt.com/), Jiahua Dong, [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), and [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ), "Xformer: Hybrid X-Shaped Transformer for Image Denoising", arXiv, 2023 

[[arXiv](https://arxiv.org/abs/2303.06440)]  [[pretrained models](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD?usp=sharing)]

### 🔥🔥🔥 News

**2024-02-01:** We open source model training and testing details, and the pretrained models can be downloaded.

**2023-10-21:** We release this repository.

---

> **Abstract:** In this paper, we present a hybrid X-shaped vision Transformer, named Xformer, which performs notably on image denoising tasks. We explore strengthening the global representation of tokens from different scopes. In detail, we adopt two types of Transformer blocks. The spatialwise Transformer block performs fine-grained local patches interactions across tokens defined by spatial dimension. The channel-wise Transformer block performs direct global context interactions across tokens defined by channel dimension. Based on the concurrent network structure, we design two branches to conduct these two interaction fashions. Within each branch, we employ an encoder-decoder architecture to capture multi-scale features. Besides, we propose the Bidirectional Connection Unit (BCU) to couple the learned representations from these two branches while providing enhanced information fusion. The joint designs make our Xformer powerful to conduct global information modeling in both spatial and channel dimensions. Extensive experiments show that Xformer, under the comparable model complexity, achieves state-of-the-art performance on the synthetic and real-world image denoising tasks.

![](figs/Xformer.jpg)

---

|                     HQ                     |                       LQ                        | [SwinIR](https://github.com/JingyunLiang/SwinIR) | [Restormer](https://github.com/swz30/Restormer) |                 Xformer (ours)                  |
| :----------------------------------------: | :---------------------------------------------: | :----------------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
| <img src="figs/ComS_img_033_HQ_N50.png" height=80> | <img src="figs/ComS_img_033_LQ_N50.png" height=80> |  <img src="figs/ComS_img_033_SwinIR_N50.png" height=80>  | <img src="figs/ComS_img_033_Restormer_N50.png" height=80> | <img src="figs/ComS_img_033_Xformer_N50.png" height=80> |
| <img src="figs/ComS_img_057_HQ_N50.png" height=80> | <img src="figs/ComS_img_057_LQ_N50.png" height=80> |  <img src="figs/ComS_img_057_SwinIR_N50.png" height=80>  | <img src="figs/ComS_img_057_Restormer_N50.png" height=80> | <img src="figs/ComS_img_057_Xformer_N50.png" height=80> |

## Installation

This repository is built in PyTorch 1.8.0. (Python3.8, CUDA11.6, cuDNN~).

1. Make conda environment
```
conda create -n pytorch18 python=3.8
conda activate pytorch18
```

2. Install dependencies
```
git clone https://github.com/gladzhang/Xformer.git
cd Xformer
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

3. Install basicsr
```
python setup.py develop --no_cuda_ext
```


## 🔗 Contents

1. [Models](#Models)
1. [Datasets](#Datasets)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [Acknowledgement](#Acknowledgement)


## Models

|  Task   | Method  | Params (M) | FLOPs (G) | Dataset  | PSNR  |  SSIM  |                          Model Zoo                           |             
| :-----: | :------ | :--------: | :-------: | :------: | :---: | :----: | :----------------------------------------------------------: | 
|   Color-DN    | Xformer   |   25.23    |   42.2   | Urban100 | 30.36 | 0.8941 | [Google Drive](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD?usp=sharing) |
|   Gray-DN    | Xformer   |   25.23  |   42.2   | Urban100 | 28.71 | 0.8629 | [Google Drive](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD?usp=sharing) | 
| Real-DN | Xformer     |   25.23   |  42.2     |   DND   | 40.19 | 0.957 | [Google Drive](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD?usp=sharing) | 

- We provide the performance Urban100 (level=50, Color-DN)  Urban100 (level=50, Gray-DN), and DND (Real-DN). We use the input 3 × 128 × 128 to calculate FLOPS.
- Download  the models and put them into the folder `experiments/pretrained_models`  . Go to the folder to find details of directory structure.

## Datasets

Used training and training sets can be downloaded as follows:

| Task                                          |                                       Training Set                          |    Testing Set  |
| :-------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| GaussionColor image denoising                                      | [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) (4744 images) | CBSD68 + Kodak24 + McMaster + Urban100 [[download]](https://drive.google.com/file/d/1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0/view?usp=sharing) |
|GaussionGrayscale image denoising                          |  [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) (4744 images)| Set12 + BSD68 + Urban100 [[download]](https://drive.google.com/file/d/1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0/view?usp=sharing)  |
| real image denoising                          | [SIDD_train](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing) |[SIDD_val](https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing) + [SIDD_test](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing) + [DND](https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing) |
| Single image motion deblurring  | [GoPro_train](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing) | [GoPro](https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/view?usp=sharing) + [HIDE](https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing) + [RealBlur_J](https://drive.google.com/file/d/1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW/view?usp=sharing) + [RealBlur_R](https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing) |


Download  training datasets and put them into the folder `datasets/`  . Go to the folder to find details of directory structure. You should go to `datasets/data_prepare` to prepare training and validating data in advance.

Download testing datasets and put them into the folder `datasets/test/`. Go to the folder to find details of directory structure.

## Training
### Train on GaussionColor image denoising
1. Please download the corresponding training datasets and put them in the folder `datasets\DFWB`.
2. Follow the instructions below to begin training our model.
```bash
# using 4 GPU with initial batch size 16 each GPU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2418 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma15.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2416 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma25.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma50.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2412 basicsr/train.py -opt options/GaussianColorDenoising_X_Former_Blind.yml --launcher pytorch
```
### Train on GaussionGrayscale image denoising
1. Please download the corresponding training datasets and put them in the folder `datasets\DFWB`.
2. Follow the instructions below to begin training our model.
```bash
# using 4 GPU with initial batch size 16 each GPU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2418 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma15.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2416 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma25.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma50.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2412 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_Former_Blind.yml --launcher pytorch
```

### Train on Real Image Denoising
1.  Please download the corresponding training datasets and put them in the folder `datasets\SIDD`. 
2. Follow the instructions below to begin training our model.
```bash
# using 4 GPU with initial batch size 16 each GPU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6414 basicsr/train.py -opt options/RealDenoising_X_Former.yml --launcher pytorch
```

### Train on Single image motion deblurring
1. Please download the corresponding training datasets and put them in the folder `datasets\GoPro`. 
2. Follow the instructions below to begin training our model.

   ```bash
   # using 4 GPU with initial batch size 16 each GPU
   python -m torch.distributed.launch --nproc_per_node=4 --master_port=2412 basicsr/train.py -opt options/MotionDeblurring_X_Former.yml --launcher pytorch
   ```

## Testing
### Test on Gaussian Color Image Denoising
1. Please download the corresponding testing datasets and put them in the folder `datasets/test/ColorDN`. Download the corresponding models and put them in the folder `experiments/pretrained_models`.
2. Follow the instructions below to begin testing our Xformer model.
```bash
#Xformer model for color image denoising. You can find corresponding results in Table 5 of the main paper.
##noise 15
python evaluate/test_gaussian_color_denoising.py --sigma 15
python evaluate/evaluate_gaussian_color_denoising.py --sigma 15
##noise 25
python evaluate/test_gaussian_color_denoising.py --sigma 25
python evaluate/evaluate_gaussian_color_denoising.py --sigma 25
##noise 50
python evaluate/test_gaussian_color_denoising.py --sigma 50
python evaluate/evaluate_gaussian_color_denoising.py --sigma 50

#Xformer for learning a single model to handle various noise levels. You can find corresponding results in Table 3 of the supplementary material.
##noise 15, 25, 50
python evaluate/test_gaussian_color_denoising.py --model_type blind --sigma 15
python evaluate/test_gaussian_color_denoising.py --model_type blind --sigma 25
python evaluate/test_gaussian_color_denoising.py --model_type blind --sigma 50
python evaluate/evaluate_gaussian_color_denoising.py
``` 

### Test on Gaussian Grayscale Image Denoising
1. Please download the corresponding testing datasets and put them in the folder `datasets/test/GrayDN`. Download the corresponding models and put them in the folder `experiments/pretrained_models`.
2. Follow the instructions below to begin testing our Xformer model.
```bash
#Xformer model for gray image denoising. You can find corresponding results in Table 4 of the main paper.
##noise 15
python evaluate/test_gaussian_gray_denoising.py --sigma 15
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 15
##noise 25
python evaluate/test_gaussian_gray_denoising.py --sigma 25
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 25
##noise 50
python evaluate/test_gaussian_gray_denoising.py --sigma 50
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 50

#Xformer for learning a single model to handle various noise levels. You can find corresponding results in Table 3 of the supplementary material.
##noise 15, 25, 50
python evaluate/test_gaussian_gray_denoising.py --model_type blind --sigma 15
python evaluate/test_gaussian_gray_denoising.py --model_type blind --sigma 25
python evaluate/test_gaussian_gray_denoising.py --model_type blind --sigma 50
python evaluate/evaluate_gaussian_gray_denoising.py
```


### Test on Real Image Denoising
1. Please download the corresponding testing datasets and put them in the folder `datasets/test/SIDD` and `datasets/test/SIDD`. Download the corresponding models and put them in the folder `experiments/pretrained_models`.
2. Follow the instructions below to begin testing our Xformer model.
```bash
#Xformer model for real image denoising. You can find corresponding results in Table 6 of the main paper.
##sidd
python evaluate/test_real_denoising_sidd.py
run evaluate_sidd.m
##dnd (You should upload the generated mat files to the online server to get PSNR and SSIM.)
python evaluate/test_real_denoising_dnd.py
```


## 🔎 Results

We achieved state-of-the-art performance on Gaussian image denoising and real-world image denoising tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expan)</summary>


- results in Table 2-4 of the main paper

<p align="center">
  <img width="900" src="figs/Results.jpg">
</p>
</details>

<details>
<summary>Visual Comparison (click to expan)</summary>



- results in Figure 4 of the main paper

<p align="center">
  <img width="900" src="figs/vis1.jpg">
</p>


- results in Figure 6 of the main paper

<p align="center">
  <img width="900" src="figs/vis2.jpg">
</p>

</details>

## 📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{zhang2023xformer,
      title={Xformer: Hybrid X-Shaped Transformer for Image Denoising}, 
      author={Jiale Zhang and Yulun Zhang and Jinjin Gu and Jiahua Dong and Linghe Kong and Xiaokang Yang},
      year={2023},
      eprint={2303.06440},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 💡 Acknowledgements

This work is released under the Apache 2.0 license.
The codes are based on [Restormer](https://github.com/swz30/Restormer) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Please also follow their licenses. Thanks for their awesome works.


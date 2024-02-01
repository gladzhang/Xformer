import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.x_former_arch import Xformer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Gasussian gray Denoising')

parser.add_argument('--input_dir', default='datasets/test/GrayDN/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/xformer_gray_dn_sigma15/', type=str, help='Directory for results')
parser.add_argument('--sigma', default='15', type=str, help='Sigma values, 15, 25, or 50')

args = parser.parse_args()

####### Load model options #######
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt_str = r"""
  type: Xformer
  inp_channels: 1
  out_channels: 1
  dim: 48
  num_blocks: [2, 4, 4]
  spatial_num_blocks: [2,4,4,6]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  window_size: [16,16,16,16]
  ffn_expansion_factor: 2.66
  bias: False
  dual_pixel_task: False
"""
opt = yaml.safe_load(opt_str)
network_type = opt.pop('type')
##########################################

sigma = np.int_(args.sigma)

factor = 8

datasets = ['Set12', 'BSD68', 'Urban100']

print("Compute results for noise level",sigma)
model_restoration = Xformer(**opt)    

if sigma == 15:
    weights = 'experiments/pretrained_models/xformer_gray_dn_sigma15.pth'
elif sigma == 25:
    weights = 'experiments/pretrained_models/xformer_gray_dn_sigma25.pth'
else:
    weights = 'experiments/pretrained_models/xformer_gray_dn_sigma50.pth'

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])

print("===>Testing using weights: ",weights)
print("------------------------------------------------")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

for dataset in datasets:
    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
    result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma))
    os.makedirs(result_dir_tmp, exist_ok=True)

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(utils.load_gray_img(file_))/255.

            np.random.seed(seed=0)  # for reproducibility
            img += np.random.normal(0, sigma/255., img.shape)

            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
            utils.save_gray_img(save_file, img_as_ubyte(restored))

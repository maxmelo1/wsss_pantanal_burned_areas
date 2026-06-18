# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--input_dir', default='resnet50@seed=0@bs=16@ep=5@nesterov@train@scale=0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--output_dir', default='make_aff_palleted', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.30, type=float)
parser.add_argument('--bg_threshold', default=0.05, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    pred_dir = f'/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/predictions/ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0/'
    aff_dir = create_directory('./experiments/predictions/make_aff_pallet_in/new/{}@aff_fg={:.2f}_bg={:.2f}/'.format(args.output_dir, args.fg_threshold, args.bg_threshold))

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU
    meta_dic = read_json('/mnt/58B0FAA2B0FA85B2/max/puzzle-cam2_rgbnir/data_queimadas_rgb/VOC_2012.json')
    #dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, 'train')
    
    #################################################################################################
    # Convert
    #################################################################################################
    eval_timer = Timer()
    
    length = len(dataset)
    for step, (ori_image, image_id, one_hot, label) in enumerate(dataset):
        png_path = aff_dir + image_id + '.png'

        cam_dict = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()

        if image_id == '2021_09_07_Planet_1_12288_11264':
            cam_test = np.expand_dims(cam_dict['hr_cam'], axis=0)
            cam_test = make_cam(torch.from_numpy(cam_test)).squeeze(0).squeeze(0)
            print(one_hot)
            print(cam_test.size())
            # input()
            import matplotlib.pyplot as plt
            
            plt.imshow(cam_test)
            plt.show()
        
        if os.path.isfile(png_path) and image_id != 'ORTOFOTO_1_8395_18677_1':
            continue

        # load
        image = np.asarray(ori_image)

        ori_h, ori_w, c = image.shape
        
        keys = cam_dict['keys']
        cams = cam_dict['hr_cam']
        cams = cams[0]
        cams_denormalized = (cams * 255).astype(np.uint8)
        cmap = cv2.COLORMAP_JET
        cams_map = cv2.applyColorMap(cams_denormalized, cmap)

        if image.shape[2] == 4:
            image = image[:, :, :3]

        cams_map = cv2.addWeighted(image, 0.5, cams_map, 0.5, 0)[..., ::-1]

        imageio.imwrite(png_path, cams_map.astype(np.uint8))

        if image_id == '2021_09_07_Planet_1_12288_11264':
            import matplotlib.pyplot as plt
            
            plt.imshow(cams)
            plt.show()

        
        sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), cams.shape))
        sys.stdout.flush()
    print()
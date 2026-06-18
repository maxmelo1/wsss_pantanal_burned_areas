import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio
from PIL import Image

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

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnest101', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

parser.add_argument('--width', type=int, default=768, help='patch width')
parser.add_argument('--height', type=int, default=768, help='patch heigth')
parser.add_argument('--step', type=int, default=384, help='step size')
parser.add_argument('--weights', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--output-path', type=str, default='./inference', help='inference output path')


if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    pred_dir = create_directory(args.output_path)

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    imagenet_mean = [0.485, 0.456, 0.406, 0.406]
    imagenet_std = [0.229, 0.224, 0.225, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    meta_dic = read_json('./data/VOC_2012.json')

    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)

    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)

    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

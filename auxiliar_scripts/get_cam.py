import os
import sys
import copy
import math
import shutil
import random
import argparse
import numpy as np

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


def generate_cams(model, image_path):
    # Load image - force RGBA mode to ensure 4 channels
    ori_image = Image.open(image_path).convert('RGBA')
    ori_w, ori_h = ori_image.size
    
    # Convert to numpy array and separate channels
    image_np = np.array(ori_image)
    rgb = image_np[..., :3]
    nir = image_np[..., 3]  # 4th channel (NIR)
    
    # Normalize RGB and NIR with your specific normalization values
    imagenet_mean_rgb = [0.041, 0.035, 0.023]
    imagenet_std_rgb = [0.019, 0.013, 0.01]
    imagenet_mean_nir = [0.126]
    imagenet_std_nir = [0.056]
    
    # Normalize RGB
    rgb = (rgb / 255.0 - imagenet_mean_rgb) / imagenet_std_rgb
    # Normalize NIR
    nir = (nir / 255.0 - imagenet_mean_nir) / imagenet_std_nir
    
    # Combine into 4-channel tensor
    image_tensor = torch.from_numpy(np.concatenate([
        rgb, 
        nir[..., np.newaxis]
    ], axis=-1)).permute(2, 0, 1).float()
    
    # Add batch dimension and move to GPU
    image_tensor = image_tensor.unsqueeze(0).cuda()
    
    # Process original image
    with torch.no_grad():
        logits, features = model(image_tensor, with_cam=True)
        features = make_cam(features)
        cam_original = features.squeeze(0).cpu().numpy()
    
    # Process tiled images (2x2 puzzle)
    num_pieces = 4
    tiled_images = tile_features(image_tensor, num_pieces)
    
    with torch.no_grad():
        tiled_logits, tiled_features = model(tiled_images, with_cam=True)
        tiled_features = make_cam(tiled_features)
        re_features = merge_features(tiled_features, num_pieces, 1)
    
    # Visualization function
    def visualize_cam(cam, image, name):
        # Take max across classes if needed
        if cam.ndim == 3:
            cam = cam.max(axis=0)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = (cam * 255).astype(np.uint8)
        
        # Resize and create heatmap
        cam = cv2.resize(cam, (ori_w, ori_h))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        
        # Prepare base image (use RGB channels only)
        image_np = np.array(image)[..., :3]
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.float32) / 255
        
        # Overlay heatmap
        overlayed = cv2.addWeighted(image_np, 0.5, heatmap.astype(np.float32)/255, 0.5, 0)
        cv2.imwrite(f'{name}_cam.jpg', overlayed * 255)
    
    # Generate visualizations
    visualize_cam(cam_original, ori_image, 'original')

if __name__ == '__main__':
    # Initialize 4-channel model
    model = Classifier('resnest101', 1, mode='normal').cuda()
    load_model(model, '/mnt/58B0FAA2B0FA85B2/max/puzzle-cam2_rgbnir/experiments_last_17_05_23/models/ResNeSt101@Puzzle@optimal.pth')
    model.eval()
    
    # Process image
    generate_cams(model, '/mnt/58B0FAA2B0FA85B2/max/puzzle-cam2_rgbnir/images_paper.png')
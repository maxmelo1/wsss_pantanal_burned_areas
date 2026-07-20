import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

try:
    from mmseg.apis import init_model, inference_model
    MMSEG_V1 = True
    print("Using MMSegmentation 1.x API")
except ImportError:
    from mmseg.apis import init_segmentor, inference_segmentor
    MMSEG_V1 = False
    print("Using MMSegmentation 0.x API")

def iou(pred, gt, eps=1e-7):
    # gt is ground truth mask with values 0 (bg) and 1 (burned)
    # pred is predicted mask with values 0 (bg) and 1 (burned)
    cal = gt < 255
    mask = (pred == gt) * cal
    tp = ((gt == 1) * mask).sum()
    fp = (cal * (gt == 0) * (pred == 1)).sum()
    fn = (cal * (gt == 1) * (pred == 0)).sum()
    
    iou_val = tp / (tp + fp + fn + eps)
    return iou_val

def main():
    parser = argparse.ArgumentParser(description="Evaluate mmseg model IoU by burned area percentage range")
    parser.add_argument('--config', type=str, default='/mnt/58B0FAA2B0FA85B2/max/Queimadas/mmsegmentation_novo/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_256x256_40k_queimadas.py', help='mmseg model config path')
    parser.add_argument('--checkpoint', type=str, default='/mnt/58B0FAA2B0FA85B2/max/Queimadas/puzzle-cam2_rgbnir/paper_queimadas/deeplabv3plus_r101-d8_256x256_40k_queimadas/best_mIoU_iter_36000.pth', help='mmseg model checkpoint path')
    parser.add_argument('--split-list', type=str, default='/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/dataset2_bkp/data/test_sem_interseccao.txt', help='split text file listing image names')
    parser.add_argument('--img-dir', type=str, default='/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/dataset2_bkp/JPEGImages', help='path to images (.npy)')
    parser.add_argument('--gt-dir', type=str, default='/mnt/58B0FAA2B0FA85B2/max/Queimadas/pantanal/queimadas/dataset2_bkp/SegmentationClass', help='path to ground truth masks (.png)')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    args = parser.parse_args()

    print("Loading model...")
    if MMSEG_V1:
        model = init_model(args.config, args.checkpoint, device=args.device)
    else:
        model = init_segmentor(args.config, args.checkpoint, device=args.device)
    print("Model loaded successfully.")

    with open(args.split_list, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(names)} files from split list.")

    MAX_SIZE = 512 * 512
    bins = [[] for _ in range(5)]
    skipped_count = 0

    print("Evaluating...")
    for name in tqdm(names):
        npy_path = os.path.join(args.img_dir, name + '.npy')
        gt_path = os.path.join(args.gt_dir, name + '.png')

        if not os.path.exists(npy_path) or not os.path.exists(gt_path):
            continue

        gt = np.array(Image.open(gt_path))
        burned = np.sum(gt == 1)

        if burned == 0:
            skipped_count += 1
            continue

        # Inference
        if MMSEG_V1:
            result = inference_model(model, npy_path)
            pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
        else:
            result = inference_segmentor(model, npy_path)
            pred = result[0]

        # Calculate percentage of burned area
        pct = (burned / MAX_SIZE) * 100
        bin_idx = int(pct // 21)
        if bin_idx > 4:
            bin_idx = 4

        # Calculate IoU
        iou_score = iou(pred, gt)
        bins[bin_idx].append(iou_score)

    print("\nEvaluation Results:")
    print(f"Skipped {skipped_count} images with 0% burned area.")
    
    keys = [f"{(el*20+1)}-{(el*20+20)}%" for el in range(5)]
    group_ious = []
    for i, key in enumerate(keys):
        if len(bins[i]) > 0:
            mean_val = np.mean(bins[i]) * 100
            group_ious.append(mean_val)
            print(f"Group {key}: {mean_val:.2f}% (count: {len(bins[i])})")
        else:
            group_ious.append(0.0)
            print(f"Group {key}: No samples")
            
    print(f"\nArray for graphic.py:")
    print(f"mmseg_iou = {np.round(group_ious, 2).tolist()}")

if __name__ == '__main__':
    main()

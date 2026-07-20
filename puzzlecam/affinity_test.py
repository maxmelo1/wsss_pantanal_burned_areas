from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument("--predict_dir", default='', type=str)
parser.add_argument('--gt_dir', default='dataset2/SegmentationClass', type=str)

parser.add_argument('--mode', default='npy', type=str) # png
parser.add_argument('--max_th', default=0.50, type=float)


args = parser.parse_args()

predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)
gt_folder = args.gt_dir



args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

output_folder = './experiments/predictions/inference/{}/'.format(args.experiment_name)

categories = ['background', 
    'river']
num_cls = len(categories)

def compare(name_list):
    for idx in range(len(name_list)):
        name = name_list[idx]

        if os.path.isfile(predict_folder + name + '.npy'):
            predict_dict = np.load(os.path.join(predict_folder, name + '.npy'), allow_pickle=True).item()
            
            if 'hr_cam' in predict_dict.keys():
                cams = predict_dict['hr_cam']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            elif 'rw' in predict_dict.keys():
                cams = predict_dict['rw']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            
            keys = predict_dict['keys']
            predict = keys[np.argmax(cams, axis=0)]
        else:
            predict = np.array(Image.open(predict_folder + name + '.png'))
        
        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file))
        
        cal = gt<255
        
        mask = (predict==gt) * cal

        

        output_folder_i = output_folder+f'1/'
        tp = (gt==1)*mask
        fp = cal*(gt==0)*(predict==1)
        fn = cal*(gt==1)*(predict==0)

        pred_img = np.zeros((predict.shape[0], predict.shape[1], 3), dtype=np.uint8)
        pred_img[tp!=0] = (255, 255, 0) #yellow  == TP
        pred_img[fp!=0] = (0, 255, 0) #green     == FP
        pred_img[fn!=0] = (0, 0, 255) #blue      == FN


        pred_img = Image.fromarray(pred_img)

        if not os.path.exists(output_folder_i):
            os.makedirs(output_folder_i, exist_ok=True)
            
        pred_img.save(output_folder_i+name+'.png')




if __name__ == '__main__':
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values

    if args.threshold is None:
            th_list = np.arange(0.05, 0.80, 0.05).tolist()
            
            best_th = 0
            best_mIoU = 0

            for th in th_list:
                args.threshold = th
                compare(name_list)
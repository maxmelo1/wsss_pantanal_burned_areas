import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import matplotlib.pyplot as plt

categories = ['background', 
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
num_cls = len(categories)


def iou(im, gt, eps=1e-7):
	cal  = gt < 255
	mask = (im==gt)*cal

	tp = (gt==1)*mask
	fp = cal*(gt==0)*(im==1)
	fn = cal*(gt==1)*(im==0)

	iou = tp.sum() / ((tp+fp+fn).sum()+eps)
	# print(iou)

	return iou

def hist(dcam, dcamrw):
	plt.style.use("ggplot")
	fig = plt.figure(1, (7,4))
	ax = fig.add_subplot(1,1,1)

	ax.plot(dcam.values(), color= 'red', label='CAM', linewidth=1.5)
	ax.plot(dcamrw.values(), color= 'blue', label='CAM+RW+DCRF', linewidth=1.5)
	ax.legend()
	plt.xlabel('# Recorte', fontsize=16)
	plt.ylabel('IoU (%)', fontsize=16)
	plt.title('IoU por recorte', fontsize=20)
	plt.show()

def compare(name_list, args):
	dcams = {}
	dcamsrw = {}
	for i, el in enumerate(name_list):
		name = el+'.png'

		# print(f'Image name: {name}')
		gt  = np.array(Image.open(os.path.join(args.gt_dir, name)))
		cam = np.array(Image.open(os.path.join(args.cam, name)))
		camrw = np.array(Image.open(os.path.join(args.cam_ref, name)))

		if gt.sum() == 0: #imagens sem queimadas não interessam pra computação aqui
			continue

		dcams[name] = iou(cam, gt)
		dcamsrw[name] = iou(camrw, gt)
		print(i)

	dcams = dict(sorted(dcams.items(), key=lambda item: item[1]))
	dcamsrw = dict(sorted(dcamsrw.items(), key=lambda item: item[1]))

	dcamsb = list(dcams.keys())[-90:-75]
	dcamsw = list(dcams.keys())[0]

	dcamsrwb = list(dcamsrw.keys())[-90:-75]
	dcamsrww = list(dcamsrw.keys())[0]

	print(f'dcams - best:')
	# print(list(dcams.keys()))
	print(dcamsb)
	print([dcams[el] for el in dcamsb])
	print(f' worst: {dcamsw} ({dcams[dcamsw]})')
	

	print(f'dcamsrw - best: {dcamsrwb}')
	print(dcamsrwb)
	print([dcamsrw[el] for el in dcamsrwb])
	print(f'worst: {dcamsrww} ({dcamsrw[dcamsrww]})')

	hist(dcams, dcamsrw)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str)


	parser.add_argument("--cam_ref", default='./predictions/AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw@crf=1', type=str)
	parser.add_argument("--cam", default='./predictions/ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10', type=str)
	parser.add_argument("--domain", default='train', type=str)
	parser.add_argument('--gt_dir', default='./dataset2_bkp/SegmentationClass', type=str)

	args = parser.parse_args()

	args.list = './data/' + args.domain + '.txt'


	df = pd.read_csv(args.list, names=['filename'])
	name_list = df['filename'].values

	print(len(name_list))
	# input()

	compare(name_list, args)
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

MAX_SIZE = 512*512


def iou(im, gt, eps=1e-7):
	cal  = gt < 255
	mask = (im==gt)*cal

	tp = (gt==1)*mask
	fp = cal*(gt==0)*(im==1)
	fn = cal*(gt==1)*(im==0)

	iou = tp.sum() / ((tp+fp+fn).sum()+eps)
	# print(iou)

	return iou

def show_hist(dcam, dcamrw):
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

	# hist = np.zeros(5, dtype=np.float32)
	# histrw = np.zeros(5, dtype=np.float32)
	hist = [[] for i in range(5)]
	histrw = [[] for i in range(5)]
	
	zeros = 0
	zerosrw = 0
	comp = 0

	for i, el in enumerate(name_list):
		name = el+'.png'
		aux = 0

		# print(f'Image name: {name}')
		gt  = np.array(Image.open(os.path.join(args.gt_dir, name)))
		cam = np.array(Image.open(os.path.join(args.cam, name)).resize((512, 512)))
		camrw = np.array(Image.open(os.path.join(args.cam_ref, name)).resize((512, 512)))		

		if gt.sum() == 0: #imagens sem queimadas não interessam pra computação aqui
			continue

		camc = np.where(cam == 255, 0, cam)
		camcrw = np.where(camrw == 255, 0, camrw)

		burned = np.sum(gt)
		burned_rw = np.sum(gt)

		print(np.unique(cam), np.unique(camrw))
		# burned = int(iou(cam, gt)*100)
		# burned_rw = int(iou(camrw, gt)*100)



		# print(np.unique(camc, return_counts=True))

		if not burned:
			zeros += 1
		else:
			if burned == MAX_SIZE:
				pos = -1
			else:
				pos = int( ((burned / MAX_SIZE)*100)  // 21)
		
			
			# print('deb', burned, MAX_SIZE, burned / MAX_SIZE, pos)
		hist[pos].append(iou(cam, gt))
		aux = hist[pos][-1]
		print(f'iou ->: {hist[pos][-1]}')

		if not burned_rw:
			zerosrw += 1
		else:
			if burned_rw == MAX_SIZE:
				pos = -1
			else:
				pos = int( ((burned_rw / MAX_SIZE)*100) // 21)
		
			
			# print((burned / MAX_SIZE)*100, pos)
		histrw[pos].append(iou(camrw, gt))
		print(f'iou ->: {histrw[pos][-1]}')

		if histrw[pos][-1]> aux:
			comp += 1
		elif histrw[pos][-1] < aux:
			comp -= 1


		dcams[name] = iou(cam, gt)
		dcamsrw[name] = iou(camrw, gt)

		print('it:', i)


	hist = [ np.array(el).mean()*100 for el in hist]
	histrw = [ np.array(el).mean()*100 for el in histrw]
	

	# hist = np.insert(hist, 0, zeros)
	# histrw = np.insert(histrw, 0, zerosrw)

	keys = [f'{(el*20+1)} - {(el*20+20)} %' for el in range(0,5)]
	# keys.insert(0, '0 %')

	X_axis = np.arange(len(keys))

	print(hist)
	print(histrw)
	print(keys)


	plt.style.use("ggplot")
	fig = plt.figure(1, (7,4))
	ax = fig.add_subplot(1,1,1)

	bars = ax.bar(X_axis-0.2, hist, 0.4, ec = "salmon", color = "#E31B23", label="PuzzleCAM")
	ax.bar(X_axis+.2, histrw, 0.4, ec = "blue", color = "#003366", label="PuzzleCAM-LU (ours)")
	
	# ax.bar_label(bars)
	plt.xticks(X_axis, keys, rotation=30, ha='right', fontsize=14)
	plt.yticks( fontsize=16)
	plt.xlabel('Burned area %', fontsize=20)
	plt.ylabel('IoU', fontsize=22)
	# plt.title('IoU by burned area %', fontsize=20)
	# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=4619.0))

	for i, val in enumerate(hist):
		ax.text(i-.35, val+.4, f"{val:.2f}", fontsize=20)
	for i, val in enumerate(histrw):
		ax.text(i+.05, val+.4, f"{val:.2f}", fontsize=20)

	plt.legend(fontsize=24)
	plt.show()

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

	show_hist(dcams, dcamsrw)


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
import rasterio
import shapefile
import numpy as np
from skimage.draw import polygon, polygon2mask
from PIL import Image, ImageDraw
from rasterio.windows import Window
import os
import geopandas as gpd
import cv2
import sys
import matplotlib.pyplot as plt
import argparse
import time

from sklearn.model_selection import train_test_split

def get_mask(img_h, img_w, shp, dataset):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    for shp_path in shp:
        data = gpd.read_file(shp_path)
        for i in range(len(data)):
            #print('%d of %d' % (i+1, len(data)))
            geoms = data.geometry[i]
            #if geoms == None: continue
            if geoms == None: 
                geoms = []
            elif geoms.type != 'MultiPolygon':
                geoms = [geoms]
            else:
                geoms = [x for x in geoms.geoms]
            for g, geom in enumerate(geoms):
                #print('%d of %d' % (g+1, len(geoms)))
                exterior_coords = [dataset.index(p[0], p[1]) for p in geom.exterior.coords[:]]
                poly = np.array([[y, x] for x, y in exterior_coords])
                cv2.fillPoly(mask, pts=[poly], color=1)
                for interior in geom.interiors:
                    poly_interior = [dataset.index(p[0], p[1]) for p in interior.coords[:]]
                    poly = np.array([[y, x] for x, y in poly_interior])
                    cv2.fillPoly(mask, pts=[poly], color=0)
    return mask

def get_image_patch(dataset, x, y, w, h):
    band_b = dataset.read(1, window=Window(y, x, w, h))
    band_g = dataset.read(2, window=Window(y, x, w, h))
    band_r = dataset.read(3, window=Window(y, x, w, h))
    band_nir = dataset.read(4, window=Window(y, x, w, h))

    MAX_VALUE = 2**14

    img = np.zeros((w, h, 4), dtype=np.uint8)
    img[:, :, 0] = (band_r / MAX_VALUE)*255
    img[:, :, 1] = (band_g / MAX_VALUE)*255
    img[:, :, 2] = (band_b / MAX_VALUE)*255
    img[:, :, 3] = (band_nir / MAX_VALUE)*255

    return img

def generate(tif, shp, out_path, slice_size, step, args):
    out_path_r_g_b = os.path.join(out_path, 'Images')
    out_path_labels = os.path.join(out_path, 'SegmentationClass')
    out_path_splits = os.path.join(out_path, 'ImageSets/Segmentation')
    filename = os.path.splitext(os.path.split(tif)[-1])[0]
    ignore_nf = args.ignore_no_feature

    
    if not os.path.isdir(out_path_r_g_b):
        os.makedirs(out_path_r_g_b, exist_ok=True)
        os.makedirs(out_path_labels, exist_ok=True)
        os.makedirs(out_path_splits, exist_ok=True)

    dataset = rasterio.open(tif)
    img_w = dataset.width
    img_h = dataset.height

    mask = get_mask(img_h, img_w, shp, dataset)

    # Image.fromarray((mask*255).astype(np.uint8)).save(filename + '_mask.png')
    filenames = []

    for x in range(0, img_h-slice_size-1, step):
        for y in range(0, img_w-slice_size-1, step):
            print('%d,%d of %d,%d' % (x, y, img_h, img_w))
            name = f'{filename}_{x}_{y}'

            if x+step >= img_h: x = img_h-step
            if y+step >= img_w: y = img_w-step      

            patch_mask = mask[x:x+slice_size, y:y+slice_size]
            patch_img = get_image_patch(dataset, x, y, slice_size, slice_size)
            invalid_pixels = np.logical_and(np.logical_and(patch_img[:,:,0]==0, patch_img[:,:,1]==0), patch_img[:,:,2]==0)
            patch_mask[invalid_pixels] = 0            

            if np.sum(invalid_pixels) > slice_size*slice_size*args.invalid_size: continue
            if np.sum(patch_mask) == 0 and ignore_nf: continue

            im_gt = Image.fromarray(patch_mask, mode='P')
            im_gt.putpalette([0, 0, 0, 0, 255, 0])
            im_gt.save(os.path.join(out_path_labels, '%s_%d_%d.png' % (filename, x, y)))
            
            # Image.fromarray(patch_img).save(os.path.join(out_path_r_g_b, '%s_%d_%d.png' % (filename, x, y)) )            
            np.save(os.path.join(out_path_r_g_b, '%s_%d_%d.npy' % (filename, x, y)), patch_img )

            filenames.append(name)
    
    
    return filenames
    

def str2bool(val: str):
    return True if val.lower() in ['true', 't', 'verdadeiro', 'v'] else False

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--slice-size', type=int, default=256, help='slice size')
    parser.add_argument('--step-size', type=int, default=128, help='overlap size')
    parser.add_argument('--val-size', type=float, default=.15, help='validation size')
    parser.add_argument('--test-size', type=float, default=.15, help='validation size')
    parser.add_argument('--output-path', type=str, default='./new_dataset/', help='output path for segmentation dataset generation')
    parser.add_argument('--image-extension', nargs='+', default='.tif')
    
    parser.add_argument('--invalid-size', type=float, default=.6, help='Maximum percentage of invalid pixels per patch.')

    args = parser.parse_args()

    args.ignore_no_feature = False

    BASE_PATH  = 'nova_versao/'
    PATH_TRAIN = ['area1', 'area3', 'Area_4', 'Area_6_tif_inteiro', 'area7_filtrado']

    slice_size = args.slice_size
    step = args.step_size
    OUT_PATH = args.output_path
    extension = args.image_extension

    if isinstance(extension, str):
        extension = [extension]

    print('Selected parameters:')
    print(args)
    
    tifs = []
    shps = []
    for path in PATH_TRAIN:
        tifs += [os.path.join(BASE_PATH, path, 'tifs', f) for f in os.listdir(os.path.join(BASE_PATH, path, 'tifs')) if f.endswith(tuple(extension))]
        shps += [os.path.join(BASE_PATH, path, 'shp', label) for label in os.listdir(os.path.join(BASE_PATH, path, 'shp')) if label.endswith('.shp')] 

        size = len(tifs)

    start_time = time.time()
    n_patches = 0
    train_file_names = []
    for i, tif in enumerate(tifs):
        print(f'{i}/{size} - Processing: {tif}\n')
        args.split = 'train'
        train_file_names += generate(tif, shps, OUT_PATH, slice_size, step, args)

    n_patches = len(train_file_names)

    train_size = int(n_patches*(1-args.val_size-args.test_size))
    val_size   = int(n_patches*(args.val_size))

    idx = np.random.permutation(n_patches)
    file_names = np.array(train_file_names)

    idx_train = idx[0:train_size]
    train_file_names = file_names[idx_train]
    with open(os.path.join(OUT_PATH, 'ImageSets/Segmentation', args.split+'.txt'), 'a') as f:
        f.write('\n'.join(train_file_names))
    
    end_time = time.time()
    print(f'processed train {n_patches} patches in {end_time-start_time} seconds')
    print('\n\n\n')

    
    # validation files
    idx_val = idx[train_size:train_size+val_size]
    val_file_names = file_names[idx_val]
    args.split = 'val'
    with open(os.path.join(OUT_PATH, 'ImageSets/Segmentation', args.split+'.txt'), 'a') as f:
            f.write('\n'.join(val_file_names))
    
    end_time = time.time()
    print(f'processed val {n_patches} patches in {end_time-start_time} seconds')
    print('\n\n\n')

    # test files
    idx_test = idx[train_size+val_size:]
    test_file_names = file_names[idx_test]
    args.split = 'test'
    with open(os.path.join(OUT_PATH, 'ImageSets/Segmentation', args.split+'.txt'), 'a') as f:
            f.write('\n'.join(test_file_names))
        
    end_time = time.time()
    print(f'processed test {n_patches} patches in {end_time-start_time} seconds')
    print('\n\n\n')


if __name__ == "__main__":
    main()

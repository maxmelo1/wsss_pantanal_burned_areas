import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from mmseg_custom import *
#from mmseg.apis import inference_model
from utils.tif import get_image_patch
from utils.shp2img import get_img
from utils.img2shp import polygons_from_binary_image
import logging
from mmseg_custom import *
from mmseg.apis import inference_segmentor, init_segmentor

# Configurando o logging
logging.basicConfig(
    filename='prediction_log.log',  # Nome do arquivo de log para a predição
    level=logging.INFO,              # Nível do log
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato da mensagem de log
)

def prediction_in_plot(gpd_talhoes, index, dataset, model, patch_size, step, min_img_size=1024, batch_size=1):
    mask, (min_x, min_y, max_x, max_y), (min_lat, max_lat, min_lon, max_lon) = get_img(gpd_talhoes, dataset.index, index=index, min_img_size=min_img_size)
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    width, height = int(max_x-min_x), int(max_y-min_y)
    
    imgs = []
    positions = []
    results_probs = np.zeros((2, width, height))

    for x in tqdm(range(min_x, max_x-1, step)):
        discard_x1, discard_x2 = 0.1, 0.9
        if x == min_x:
            discard_x1 = 0.
        if x+patch_size >= max_x:
            x = max_x-patch_size
            discard_x2 = 1.
        for y in range(min_y, max_y-1, step):
            discard_y1, discard_y2 = 0.1, 0.9
            if y+patch_size >= max_y:
                discard_y2 = 1.
                y = max_y-patch_size
            if y == min_y:
                discard_y1 = 0.

            x1 = (x-min_x)+int(patch_size*discard_x1)
            y1 = (y-min_y)+int(patch_size*discard_y1)
            x2 = (x-min_x)+int(patch_size*discard_x2)
            y2 = (y-min_y)+int(patch_size*discard_y2)

            mask_patch = mask[x1:x2, y1:y2]
            if np.sum(mask_patch) == 0:
                continue

            img = get_image_patch(dataset, x, y, patch_size, patch_size)
            if img is None:
                continue
            
            img = img[:, :, [2, 1, 0]]
            imgs.append(img)
            positions.append([x1, x2, y1, y2, discard_x1, discard_x2, discard_y1, discard_y2, x, y])
            if len(imgs) >= batch_size:
                for img_i, pos_i in zip(imgs, positions):
                    results_all = inference_segmentor(model, img_i)
                    x1, x2, y1, y2, discard_x1, discard_x2, discard_y1, discard_y2, x, y = pos_i
                    patch_daninha = results_all[0]
                    patch_daninha = patch_daninha[int(patch_size*discard_x1):int(patch_size*discard_x2),
                              int(patch_size*discard_y1):int(patch_size*discard_y2)]
                    results_probs[0, x1:x2, y1:y2] += (patch_daninha == 0)
                    results_probs[1, x1:x2, y1:y2] += (patch_daninha == 1)
                imgs = []
                positions = []


    if len(imgs) > 0:
        for img_i, pos_i in zip(imgs, positions):
            results_all = inference_segmentor(model, img_i)
            x1, x2, y1, y2, discard_x1, discard_x2, discard_y1, discard_y2, x, y = pos_i
            patch_daninha = results_all[0]
            patch_daninha = patch_daninha[int(patch_size*discard_x1):int(patch_size*discard_x2),
                      int(patch_size*discard_y1):int(patch_size*discard_y2)]
            results_probs[0, x1:x2, y1:y2] += (patch_daninha == 0)
            results_probs[1, x1:x2, y1:y2] += (patch_daninha == 1)

        results_all = inference_segmentor(model, imgs)

        for i in range(len(results_all)):
            x1, x2, y1, y2, discard_x1, discard_x2, discard_y1, discard_y2, x, y = positions[i]
            patch_daninha = results_all[0]
            patch_daninha = patch_daninha[int(patch_size*discard_x1):int(patch_size*discard_x2),
                              int(patch_size*discard_y1):int(patch_size*discard_y2)]
            results_probs[0, x1:x2, y1:y2] += (patch_daninha == 0)
            results_probs[1, x1:x2, y1:y2] += (patch_daninha == 1)

    #################
    #To X-Large otophotos
    #split_idx = height // 2
    #results_top_half = np.argmax(results_probs[:, :, :split_idx], axis=0).astype(np.uint8)
    #results_bottom_half = np.argmax(results_probs[:, :, split_idx:], axis=0).astype(np.uint8)
    #results = np.concatenate((results_top_half, results_bottom_half), axis=1)
    ##################
  
    results = np.argmax(results_probs, axis=0).astype(np.uint8)  
    
    if mask is not None:
        results[mask == 0] = 0

    results_shp = polygons_from_binary_image(results, dataset.transform, dataset.crs, min_x=min_x, min_y=min_y)
    return results, results_shp
    
def prediction(shp_path, tif_path, model, patch_size, step,class_name):
    gpd_mask = gpd.read_file(shp_path)
    dataset = rasterio.open(tif_path)
    gpd_mask = gpd_mask.to_crs(dataset.crs)
    shp_all_mask = []

    for i, index in enumerate(range(len(gpd_mask))):
        print(f'Processando {class_name}: {i+1}/{len(gpd_mask)}')
        _, results_shp = prediction_in_plot(gpd_mask, index, dataset, model, 512, step)
        #input()
        shp_all_mask.append(results_shp)
    # Concatenando uma lista de GeoDataFrames
    gdf_all_mask = gpd.GeoDataFrame(pd.concat(shp_all_mask, ignore_index=True), crs=dataset.crs)
    return gdf_all_mask

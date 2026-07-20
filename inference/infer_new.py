import os
import numpy as np
from PIL import Image, ImageDraw
import time
import rasterio
import geopandas as gpd
import cv2
import glob
import gc
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from mmseg_custom import *
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

def get_bands(tif):
    """
    Carrega as bandas da imagem TIF e normaliza os valores.
    """
    MAX_VAL = 2**14 + 0.00
    dataset = rasterio.open(tif)
    band_b, band_g, band_r = dataset.read(1), dataset.read(2), dataset.read(3)
    
    not_valid_0 = np.logical_and(np.logical_and(band_r == 0, band_g == 0), band_b == 0)
    valid = np.logical_not(not_valid_0)
    
    band_r = (band_r / MAX_VAL) * 255
    band_g = (band_g / MAX_VAL) * 255
    band_b = (band_b / MAX_VAL) * 255

    return band_r, band_g, band_b, dataset, valid

def polygons_from_binary_image(binary_image, transform, crs, min_x, min_y):
    """
    Converte uma imagem binária (máscara) em polígonos GeoPandas.
    Esta é uma versão simplificada para este exemplo.
    """
    from shapely.geometry import Polygon
    
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    if hierarchy is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    hierarchy = hierarchy[0]
    
    for i in range(len(contours)):
        if hierarchy[i][3] == -1: # Contornos externos
            cnt = contours[i]
            if len(cnt) < 3:
                continue

            coords = []
            for point in cnt.squeeze():
                x_px, y_px = point[0], point[1]
                lon, lat = rasterio.transform.xy(transform, y_px + min_y, x_px + min_x)
                coords.append((lon, lat))

            # Verifica contornos internos (buracos)
            interiors = []
            child = hierarchy[i][2]
            while child != -1:
                child_cnt = contours[child]
                child_coords = []
                for point in child_cnt.squeeze():
                    x_px, y_px = point[0], point[1]
                    lon, lat = rasterio.transform.xy(transform, y_px + min_y, x_px + min_x)
                    child_coords.append((lon, lat))
                interiors.append(child_coords)
                child = hierarchy[child][0]

            if interiors:
                polygons.append(Polygon(coords, interiors))
            else:
                polygons.append(Polygon(coords))
    
    return gpd.GeoDataFrame(geometry=polygons, crs=crs)

def prediction_in_plot(gpd_plot, plot_index, dataset, model, patch_size, step, batch_size):
    """
    Processa um único talhão/plot, aplicando a segmentação.
    """
    # Recorta a imagem para o bounding box do talhão
    minx, miny, maxx, maxy = gpd_plot.geometry.loc[plot_index].bounds
    min_x_px, min_y_px = dataset.index(minx, maxy)
    max_x_px, max_y_px = dataset.index(maxx, miny)

    min_x_px, min_y_px = int(min_x_px), int(min_y_px)
    max_x_px, max_y_px = int(max_x_px), int(max_y_px)

    width = max_x_px - min_x_px
    height = max_y_px - min_y_px

    if width <= 0 or height <= 0:
        print(f"Dimensões do plot {plot_index} inválidas.")
        return None

    # Inicializa arrays com as dimensões corretas do bounding box
    imgs = []
    positions = []
    results_probs = np.zeros((2, height, width), dtype=np.float32)

    band_r, band_g, band_b, _, _ = get_bands(dataset.name)

    # Note: the loops now iterate over the *height* (x-dim) and *width* (y-dim) of the plot
    for x in tqdm(range(min_x_px, max_x_px, step)):
        if x + patch_size > max_x_px:
            x = max_x_px - patch_size

        for y in range(min_y_px, max_y_px, step):
            if y + patch_size > max_y_px:
                y = max_y_px - patch_size

            img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            img[:, :, 0] = band_r[x:x+patch_size, y:y+patch_size]
            img[:, :, 1] = band_g[x:x+patch_size, y:y+patch_size]
            img[:, :, 2] = band_b[x:x+patch_size, y:y+patch_size]

            img = img[:, :, [2, 1, 0]]
            
            imgs.append(img)
            x_rel, y_rel = x - min_x_px, y - min_y_px
            positions.append([x_rel, y_rel])
            
            if len(imgs) >= batch_size:
                results_all = [inference_segmentor(model, img_batch) for img_batch in imgs]
                for i in range(len(results_all)):
                    x_rel, y_rel = positions[i]
                    # Ensure the patch_probs is correctly sliced from the results
                    patch_probs = F.softmax(torch.from_numpy(results_all[i][0]).float(), dim=0).cpu().numpy()
                    results_probs[:, x_rel:x_rel+patch_size, y_rel:y_rel+patch_size] += patch_probs
                imgs = []
                positions = []

    if len(imgs) > 0:
        results_all = [inference_segmentor(model, img_batch) for img_batch in imgs]
        for i in range(len(results_all)):
            x_rel, y_rel = positions[i]
            patch_probs = F.softmax(torch.from_numpy(results_all[i][0]).float(), dim=0).cpu().numpy()
            results_probs[:, x_rel:x_rel+patch_size, y_rel:y_rel+patch_size] += patch_probs

    final_mask = np.argmax(results_probs, axis=0).astype(np.uint8)

    # Cria a máscara do talhão com as mesmas dimensões
    plot_mask = np.zeros((height, width), dtype=np.uint8)
    plot_geometry = gpd_plot.geometry.loc[plot_index]

    if plot_geometry.type != 'MultiPolygon':
        geoms = [plot_geometry]
    else:
        geoms = [x for x in plot_geometry.geoms]

    for geom in geoms:
        exterior_coords = [dataset.index(p[0], p[1]) for p in geom.exterior.coords[:]]
        poly = np.array([[y-min_y_px, x-min_x_px] for x, y in exterior_coords])
        cv2.fillPoly(plot_mask, pts=[poly], color=1)
        for interior in geom.interiors:
            poly_interior = [dataset.index(p[0], p[1]) for p in interior.coords[:]]
            poly = np.array([[y-min_y_px, x-min_x_px] for x, y in poly_interior])
            cv2.fillPoly(plot_mask, pts=[poly], color=0)

    # Agora as dimensões devem ser compatíveis
    final_mask[plot_mask == 0] = 0

    results_shp = polygons_from_binary_image(final_mask, dataset.transform, dataset.crs, min_x=min_x_px, min_y=min_y_px)
    return results_shp

def main(img_path, shp_path):
    model_path = '/mnt/58B0FAA2B0FA85B2/bauce/InternImage/segmentation/work_dirs/talhoes/focus-2/best_mIoU_iter_24000.pth'
    model_config = '/mnt/58B0FAA2B0FA85B2/bauce/InternImage/segmentation/work_dirs/talhoes/focus-2/segformer_internimage_xl_512x512_160k_talhoes.py'
    
    print(f'Gerando shapefiles para: {img_path}')
    
    model = init_segmentor(model_config, model_path, device='cuda:0')
    patch_size = 512
    step = 256
    batch_size = 1  # Ajuste conforme a memória da sua GPU
    
    # Carrega a imagem e o shapefile de talhões
    dataset = rasterio.open(img_path)
    gpd_plots = gpd.read_file(shp_path)
    gpd_plots = gpd_plots.to_crs(dataset.crs)
    
    all_predicted_polygons = []
    
    print(f'Processando {len(gpd_plots)} talhões...')
    for i, plot in gpd_plots.iterrows():
        print(f'-> Processando talhão {i+1} de {len(gpd_plots)}')
        shp = prediction_in_plot(gpd_plots, i, dataset, model, patch_size, step, batch_size)
        if shp is not None:
            all_predicted_polygons.append(shp)
            
    if all_predicted_polygons:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_predicted_polygons, ignore_index=True), crs=dataset.crs)
        
        output_dir = 'inference'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(output_dir, f'{filename}_prediction_internimage.shp')
        final_gdf.to_file(output_path)
        print(f'Shapefile final salvo em: {output_path}')
    else:
        print("Nenhum polígono foi gerado.")

if __name__ == "__main__":
    gc.enable()

    base_path = '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test'
    fazendas = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    fazendas = ['/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/100077', 
                '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/10613 - SANTO ANTONIO DA LIBERDADE'] 
    class_name = "talhoes"

    for fazenda in fazendas:
        arquivos = os.listdir(fazenda)
        path_tif = ''
        path_shape = ''

        for arquivo in arquivos:
            if arquivo.endswith('.tif'):
                path_tif = os.path.join(fazenda, arquivo)
            elif arquivo.endswith('.shp'):
                nome_camada, _ = os.path.splitext(arquivo)
                partes = nome_camada.split('_')
                if ('maskt' in partes and class_name == 'arvores') or ('mascara' in partes and class_name == 'talhoes'):
                    path_shape = os.path.join(fazenda, arquivo)

        if not path_tif or not path_shape:
            print(f"[AVISO] Não encontrei tif ou shp válido em {fazenda}, pulando...")
            continue
        
        print(f"Processando fazenda: {fazenda}")
        main(path_tif, path_shape)
        
    print("Processamento concluído.")
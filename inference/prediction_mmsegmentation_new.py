import os
import argparse
from easydict import EasyDict as edict
from models.load import get_mmsegmentation_model
from utils.files import find_subfolders_in_folder, find_tif_shp_in_folder
from utils.shp2img import get_img
from prediction.prediction_orthophoto import prediction
import sys
import torch
from mmseg_custom import *
from mmseg.apis import inference_segmentor, init_segmentor

def main(config_path):
    config_module = __import__(args.config.replace('.py', ''))
    cfg = config_module.config

    cfg_file = cfg.config_file
    chkpt = cfg.checkpoint_file
    path_file = cfg.path_folder
    patch_size = cfg.patch_size
    step = patch_size //2
    class_name = args.class_name

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model = init_segmentor(cfg_file, chkpt, device)

    orto_paths = [os.path.join(path_file, path) for path in os.listdir(path_file)]
    orto_paths = ['/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/0791', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/100077', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/100084-013', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/10040-Orto-Ecw', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/10137-10720-Orto-Ecw', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/1056-10718-Ecw', 
                  #'/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/10613 - SANTO ANTONIO DA LIBERDADE', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/10736-Orto-Ecw', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/2014-2015', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/30438 - AROEIRAS', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/50413-Orto-010', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/50438-Orto-002', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/50439-Orto-019', 
                  '/mnt/58B0FAA2B0FA85B2/bauce/Dataset/Arvores/test/50442 - MGA'
                  ]

    print("---------------------------- Iniciando processamento ---------------------------- ")
    print(f"Total de ortofotos a serem processadas: {len(orto_paths)}\n")
    print(orto_paths)
    for o, orto_path in enumerate(orto_paths):
        print(f'Processando ortofoto: {(o+1)}/{len(orto_paths)}: {orto_path}')
        # Caminho da ortofoto
        nome_camada, _ = os.path.splitext(orto_path)
        partes = nome_camada.split('/')
        filename_orto = partes[-1]
        arquivos = os.listdir(orto_path)
      
        path_shp_talhoes = ''
        path_shp_arvores = ''
        path_shp_mascara = ''
        path_shp_mask_trees = ''
        path_tif = ''

        for arquivo in arquivos:
            if arquivo.endswith('.shp'):
                nome_camada, _ = os.path.splitext(arquivo)
                partes = nome_camada.split('_')

                if 'talhoes' in partes:
                    path_shp_talhoes = os.path.join(orto_path, arquivo)
                elif 'arvores' in partes:
                    path_shp_arvores = os.path.join(orto_path, arquivo)
                elif 'mascara' in partes:
                    path_shp_mascara = os.path.join(orto_path, arquivo)
                elif 'maskt' in partes:
                    path_shp_mask_trees = os.path.join(orto_path, arquivo)

            elif arquivo.endswith('.tif'):
                path_tif = os.path.join(orto_path, arquivo)

            if class_name == 'talhoes':
                path_shape = path_shp_mascara
            elif class_name == 'arvores':
                path_shape = path_shp_mask_trees
            
        print(path_tif, '\n', path_shape)
        shp = prediction(path_shape, path_tif, model, patch_size, step,class_name)
        shp.to_file(os.path.join(orto_path, f'./{filename_orto}_prediction_{class_name}_internimage.shp'))

        #evaluation(orto_path,class_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ortho photos using a segmentation model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--class_name', type=str, default='talhoes', required=False, help='Which Class will predict')

    args = parser.parse_args()
    main(args.config)

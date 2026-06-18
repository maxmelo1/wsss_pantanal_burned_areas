import numpy as np
from PIL import Image
import os

def normalize_to_255(array):
    array = array.astype(np.float32)
    for i in range(array.shape[2]):
        ch_min = array[..., i].min()
        ch_max = array[..., i].max()
        if ch_max > ch_min:
            array[..., i] = (array[..., i] - ch_min) / (ch_max - ch_min) * 255
        else:
            array[..., i] = 0  # canal constante
    return array.astype(np.uint8)

def npy_to_png(npy_path):
    array = np.load(npy_path)

    array = normalize_to_255(array)
    img = Image.fromarray(array, mode='RGBA')
    
    print(img)

    img.save('saida.png')

# Carrega o array
array = '/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/JPEGImages/2021_09_07_Planet_1_8192_4096.npy'

npy_to_png(array)




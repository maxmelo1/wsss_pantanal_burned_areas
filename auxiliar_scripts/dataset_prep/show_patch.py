from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_PATH = '/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/JPEGImages'
OUTPUT_PATH = './converted/'

def convert(name):
    img = np.load(os.path.join(BASE_PATH, name+'.npy'))

    perc = np.zeros((4,2))
    perc[0] = np.percentile(img[..., 0], (2,98))
    perc[1] = np.percentile(img[..., 1], (2,98))
    perc[2] = np.percentile(img[..., 2], (2,98))
    perc[3] = np.percentile(img[..., 3], (2,98))


    res = (img.astype(float) - perc[...,0])/ (perc[...,1] - perc[...,0])
    res = np.maximum(np.minimum(res*255, 255), 0).astype(np.uint8)

    print(np.sum(res > 255.))

    res = res[..., :3]
    rgb = res[..., ::-1]
    Image.fromarray(res).save(os.path.join(OUTPUT_PATH, 'bgr', name+'.png'))
    Image.fromarray(rgb).save(os.path.join(OUTPUT_PATH, 'rgb', name+'.png'))
    # plt.imshow(res[..., ::-1])
    # plt.show()


with open('/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/data/test.txt', 'r') as file:
    fnames =  [line.rstrip() for line in file]


print(f'Loaded {len(fnames)} images.')

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(os.path.join(OUTPUT_PATH, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'bgr'), exist_ok=True)

for name in fnames:
    convert(name)




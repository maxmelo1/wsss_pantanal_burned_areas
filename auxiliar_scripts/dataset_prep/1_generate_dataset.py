import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image, ImageDraw
import shapefile
import cv2
import os
from skimage.morphology import disk, binary_dilation, convex_hull_image
from mmseg.apis import init_segmentor, inference_segmentor





def get_image_patch(dataset, x, y, w, h, max_value=2.**14):
    img = np.zeros((w, h, 4), dtype=np.uint16)
    
    nir = dataset.read(4, window=Window(y, x, w, h))
    red = dataset.read(3, window=Window(y, x, w, h)) # rgb - bgr
    green = dataset.read(2, window=Window(y, x, w, h))
    blue = dataset.read(1, window=Window(y, x, w, h))
    ndvi = (nir - red) / (nir + red)
    value = 255

    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue
    img[:, :, 3] = nir

    img[img>max_value] = max_value

    # if np.sum(img > 1) > 0:
    #     print('Erro')
    #     exit(0)


    return ((img / max_value)*255).astype(np.uint8)

def create_image(tif, shps, areas, path_out, s = 1024, step=None):
    filename = os.path.splitext(os.path.basename(tif))[0]
    print(tif)
    dataset = rasterio.open(tif)
    w = dataset.width
    h = dataset.height

    print('width: %d, height: %d' % (w, h))

    if step is None:
        step = s
    
    img_lines = np.zeros((h,w), np.uint8)
    img_areas = np.zeros((h,w), np.uint8)
    for f in shps:
        myshp_linhas = open(f, "rb")
        mydbf_linhas = open(f.replace('.shp', '.dbf'), "rb")
        lines = shapefile.Reader(shp=myshp_linhas, dbf=mydbf_linhas)
        shapes_linhas = lines.shapes()
        for shape in shapes_linhas:
            parts = shape.parts
            for lp in range(len(parts)):
                fi = parts[lp+1] if lp < len(parts)-1 else len(shape.points)
                points = shape.points[parts[lp]:fi]
                # points = shape.points
                # ponto -> coord geo. 
                for j, p in enumerate(points):
                    row, col = dataset.index(p[0], p[1])
                    if j > 0:
                        pj1 = points[j-1]
                        row1, col1 = dataset.index(pj1[0], pj1[1])
                        img_lines = cv2.line(img_lines, (int(col1), int(row1)),(int(col),int(row)), 1, 1)
    for f in areas:
        myshp_linhas = open(f, "rb")
        mydbf_linhas = open(f.replace('.shp', '.dbf'), "rb")
        lines = shapefile.Reader(shp=myshp_linhas, dbf=mydbf_linhas)
        shapes_linhas = lines.shapes()
        for shape in shapes_linhas:
            parts = shape.parts
            for lp in range(len(parts)):
                fi = parts[lp+1] if lp < len(parts)-1 else len(shape.points)
                points = shape.points[parts[lp]:fi]
                polygon = []
                for j, p in enumerate(points):
                    row, col = dataset.index(p[0], p[1])
                    polygon.append((col, row))
                img_areas = cv2.fillPoly(img_areas, pts = [np.array(polygon)], color=1)

    path_out_rgb = os.path.join(path_out, 'img')
    path_out_rgb2 = os.path.join(path_out, 'img_ch')
    
    path_out_shp_lines = os.path.join(path_out, 'ann')
    path_out_shp_dil_lines = os.path.join(path_out, 'ann_dil')

    #print(np.sum(cv2.dilate(img_lines, element)))
    #Image.fromarray(cv2.dilate(img_lines, element)*255).save(os.path.join(path_out, filename + '_estradas.png'))
    #Image.fromarray(img_areas*255).save(os.path.join(path_out, filename + '_area.png'))
    
    if not os.path.isdir(path_out_rgb):
        os.makedirs(path_out_rgb)
        os.makedirs(path_out_rgb2)
        os.makedirs(path_out_shp_lines)
        os.makedirs(path_out_shp_dil_lines)

    for xi in range(0, h-s-1, step):
        for yi in range(0, w-s-1, step):
            #print('%d,%d' % (xi,yi))
            img_shp_patch_lines = img_lines[xi:xi+s, yi:yi+s]
            nn = np.sum(img_shp_patch_lines == 1)
            # print(nn)
            if nn < 5:
                continue

            img = get_image_patch(dataset, xi, yi, s, s)
            if np.sum(img == 0) > 0.8*img.shape[0]*img.shape[1]*img.shape[2]:
                continue

            img_shp_patch_dil_lines = cv2.dilate(img_shp_patch_lines, element)

            im_gt_lines = Image.fromarray(img_shp_patch_lines, mode='P')
            im_gt_lines.putpalette([0, 0, 0, 0, 255, 0])
            im_gt_dil_lines = Image.fromarray(img_shp_patch_dil_lines, mode='P')
            im_gt_dil_lines.putpalette([0, 0, 0, 0, 255, 0])

            convex_hull = convex_hull_image(np.array(im_gt_dil_lines))
            convex_hull = cv2.dilate((convex_hull*255).astype(np.uint8), element)
            img2 = img.copy()
            img2[np.logical_not(convex_hull)] = [0, 0, 0, 0] 

            np.save(os.path.join(path_out_rgb, '%s_image_%d_%d' % (filename, xi, yi)), img)
            np.save(os.path.join(path_out_rgb2, '%s_image_%d_%d' % (filename, xi, yi)), img2)
            im_gt_lines.save(os.path.join(path_out_shp_lines, '%s_image_%d_%d.png' % (filename, xi, yi)))
            im_gt_dil_lines.save(os.path.join(path_out_shp_dil_lines, '%s_image_%d_%d.png' % (filename, xi, yi)))

path_ann = './novo_ds/shapefiles_fixed'
path_areas = './novo_ds/areas_fixed'
path_img = './novo_ds/tifs'
s = 256
step = None
size_dilation=10
path_out = './dataset_wh_%d_step_modificado/' % (s)

with open('./novo_ds/ImageSets/Segmentation/test.txt', 'r') as f:
    test_tifs = []
    for line in f:
        test_tifs.append(line.rstrip())




tifs = [os.path.join(path_img, f) for f in os.listdir(path_img) if f.endswith('.tif') and f.split('/')[-1][:-4] not in test_tifs]
shps = [os.path.join(path_ann, f) for f in os.listdir(path_ann) if  f.endswith('.shp')]
areas = [os.path.join(path_areas, f) for f in os.listdir(path_areas) if  f.endswith('.shp')]

tifs.remove('./novo_ds/tifs/L15-0695E-0911N.tif')

print(len(tifs), len(shps), len(areas), shps[0], areas[0])
# input()

for tif in tifs:
    create_image(tif, shps, areas, path_out, s, size_dilation, step)
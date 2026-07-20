from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import shapefile
import geopandas
from PIL import Image, ImageDraw

import cv2


def save_shapefile(filename, results, dataset):
    contours, _ = cv2.findContours(results, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        cnt = np.squeeze(cnt)
        print(cnt.shape)
        # if cnt.shape[0] <= 5: continue
        poly = []
        for x,y in cnt:
            position = [dataset.xy(y, x)]
            poly.append((position[0][0], position[0][1]))
        polygons.append(poly)

    w = shapefile.Writer(filename)
    w.field('name', 'C')
    w.poly(polygons)
    w.record('swimming pools')
    w.close()

def save_shapefile_internal(filename, results, dataset):
    from shapely.geometry import Point, Polygon

    newdata = geopandas.GeoDataFrame()
    newdata['geometry'] = None
    contours, hierarchy = cv2.findContours(results, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    polygons_internal = []
    index = 0
    new_img = Image.fromarray(np.zeros(results.shape, dtype=np.uint8))
    draw = ImageDraw.Draw(new_img)
    num = 0
    while index != -1:
        #print(index)
        cnt = contours[index]
        cnt = np.squeeze(cnt)
        if cnt.shape[0] <= 5:
            index = hierarchy[0, index, 0]
            continue
        poly_ext = []
        points = []
        for x,y in cnt:
            points.append((x, y))
            position = [dataset.xy(y, x)]
            poly_ext.append((position[0][0], position[0][1]))
        polygons.append(poly_ext)
        draw.polygon((points), fill=255)
        child = hierarchy[0, index, 2]
        internals = []
        if child != -1:
            while child != -1:
                cnt = contours[child]
                cnt = np.squeeze(cnt)
                poly = []
                points = []
                for x,y in cnt:
                    points.append((x, y))
                    position = [dataset.xy(y, x)]
                    poly.append((position[0][0], position[0][1]))
                internals.append(poly)
                draw.polygon((points), fill=0)
                child = hierarchy[0, child, 0]
        if len(internals) > 0:
            newdata.loc[num, 'geometry'] = Polygon(poly_ext, internals)
        else:
            newdata.loc[num, 'geometry'] = Polygon(poly_ext)
        num += 1
        polygons_internal.append(internals)
        index = hierarchy[0, index, 0]
    # new_img.save('polygons.png')
    # Create an empty geopandas GeoDataFrame
    #newdata.crs = from_epsg(4326)
    # Determine the output path for the Shapefile
    # Write the data into that Shapefile
    print('-->',filename)
    newdata.crs = dataset.crs
    newdata.to_file(filename)

# img = '/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/original_data/areas/Area_6/tifs/2021_09_07_Planet_2.tif'
img = '/mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/original_data/areas/Area_3/tifs/20210928_131659_63_242d_3B_AnalyticMS_SR.tif'

dataset = rasterio.open(img)

img_w = dataset.width
img_h = dataset.height

step = 512

mask = np.zeros((img_h, img_w), dtype=np.uint8)

x_c = 2560
y_c = 1024


for x in range(0, img_h-1, step):
    for y in range(0, img_w-1, step):
        print('%d,%d of %d,%d' % (x, y, img_h, img_w))
        if x+step >= img_h: x = img_h-step
        if y+step >= img_w: y = img_w-step

        if y_c ==y and x_c == x:
            mask[x:x+step, y:y+step] = 255


Image.fromarray(mask).save('20210928_131659_63_242d_3B_AnalyticMS_SR' + '_mask.png')
save_shapefile('20210928_131659_63_242d_3B_AnalyticMS_SR' + '_mask.shp', mask, dataset=dataset)



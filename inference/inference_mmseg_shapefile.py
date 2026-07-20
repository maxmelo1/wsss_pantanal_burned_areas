import os
import glob
import argparse
import logging
import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
import cv2
import shapely.geometry
from tqdm import tqdm

# Silence warnings and GDAL logs
warnings.filterwarnings("ignore")
os.environ["GDAL_PAM_ENABLED"] = "NO"
os.environ["CPL_LOG_ERRORS"] = "OFF"
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('pyproj').setLevel(logging.ERROR)
logging.getLogger('fiona').setLevel(logging.ERROR)

# Import MMSegmentation 1.x APIs
from mmseg.apis import init_model, inference_model

# Config logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def to_numpy2(transform):
    return np.array([transform.a,
                     transform.b,
                     transform.c,
                     transform.d,
                     transform.e,
                     transform.f, 0, 0, 1], dtype='float64').reshape((3, 3))


def xy_np(transform, rows, cols, min_x, min_y, offset='center'):
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows+min_y, cols+min_x, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows + min_y
        pts[1] = cols + min_x
    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")
    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    lat, lon = locs[0].tolist(), locs[1].tolist()
    coords = [(lat[i], lon[i]) for i in range(len(lat))]
    return coords


def polygons_from_binary_image(img, transform, crs, min_x=0, min_y=0, min_area=5):
    assert isinstance(img, np.ndarray), 'img deve ser um numpy array'
    unique_values = np.unique(img)
    new_geo_data_frame = {"geometry": [], 'CLASSE': []}

    for cat in unique_values:
        if cat == 0:
            continue

        img_ = img.copy()
        img_[img_ != cat] = 0
        img_[img_ != 0] = 1

        contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        categ = cat

        interiors = [xy_np(transform, contour[:, 0, 0], contour[:, 0, 1], min_x, min_y)
                     for c, contour in enumerate(contours) if hierarchy[0][c][3] != -1]
        index = [
            hierarchy[0][c][3]
            for c, contour in enumerate(contours)
            if hierarchy[0][c][3] != -1
        ]

        for c, contour in enumerate(contours):
            if hierarchy[0][c][3] == -1:
                if cv2.contourArea(contour) < min_area:
                    continue
                exterior = xy_np(
                    transform, contour[:, 0, 0], contour[:, 0, 1], min_x, min_y)
                interior = [hole for h, hole in enumerate(
                    interiors) if index[h] == c]
                if len(exterior) <= 3:
                    continue
                poly = shapely.geometry.polygon.Polygon(exterior, interior)
                new_geo_data_frame["geometry"].append(poly)
                new_geo_data_frame["CLASSE"].append(categ)

    gdf1 = gpd.GeoDataFrame.from_dict(new_geo_data_frame, geometry="geometry", crs=crs)
    return gdf1


def get_img(geo_data_frame, transform, index, min_img_size):
    bounds = geo_data_frame.iloc[[index]].bounds
    min_lat = bounds["minx"].min()
    max_lon = bounds["miny"].min()
    max_lat = bounds["maxx"].max()
    min_lon = bounds["maxy"].max()

    inv_transform = ~transform
    
    min_col, min_row = inv_transform * (min_lat, min_lon)
    min_col, min_row = int(np.floor(min_col)), int(np.floor(min_row))
    
    max_col, max_row = inv_transform * (max_lat, max_lon)
    max_col, max_row = int(np.ceil(max_col)), int(np.ceil(max_row))

    min_x, min_y = min_row, min_col
    max_x, max_y = max_row, max_col

    if not ((max_x - min_x > 0) and (max_y - min_y > 0)):
        logger.warning(f"Feature {index} has invalid/empty size.")
        return None

    if max_x - min_x < min_img_size:
        max_x = min_x + min_img_size
    if max_y - min_y < min_img_size:
        max_y = min_y + min_img_size

    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    width, height = int(max_x - min_x), int(max_y - min_y)

    image = np.zeros((width, height), dtype=np.uint8)
    geometry = geo_data_frame.geometry.iloc[index]

    # Vectorized conversion for coordinates
    coords = np.array(geometry.exterior.coords)
    cols, rows = inv_transform * (coords[:, 0], coords[:, 1])
    pts = np.column_stack((cols, rows))
    pts = np.round(pts).astype(np.int32)
    pts = pts - np.array([min_y, min_x])
    cv2.fillPoly(image, [pts], 1)

    for interior in geometry.interiors:
        coords = np.array(interior.coords)
        cols, rows = inv_transform * (coords[:, 0], coords[:, 1])
        pts = np.column_stack((cols, rows))
        pts = np.round(pts).astype(np.int32)
        pts = pts - np.array([min_y, min_x])
        cv2.fillPoly(image, [pts], 0)

    return (
        image,
        (min_x, min_y, max_x, max_y),
        (min_lat, max_lat, min_lon, max_lon),
    )


def get_image_patch(dataset, x, y, w, h, max_value=2**14):
    window = Window(y, x, w, h)
    band_b = dataset.read(1, window=window, boundless=True, fill_value=0)
    band_g = dataset.read(2, window=window, boundless=True, fill_value=0)
    band_r = dataset.read(3, window=window, boundless=True, fill_value=0)
    band_nir = dataset.read(4, window=window, boundless=True, fill_value=0)

    img = np.zeros((w, h, 4), dtype=np.uint8)
    img[:, :, 0] = np.clip((band_r / max_value) * 255.0, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip((band_g / max_value) * 255.0, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip((band_b / max_value) * 255.0, 0, 255).astype(np.uint8)
    img[:, :, 3] = np.clip((band_nir / max_value) * 255.0, 0, 255).astype(np.uint8)
    return img


def prediction_in_plot(gpd_talhoes, index, dataset, model, patch_size, step, min_img_size=1024):
    res = get_img(gpd_talhoes, dataset.transform, index=index, min_img_size=min_img_size)
    if res is None:
        return None, None
    mask, (min_x, min_y, max_x, max_y), _ = res
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    width, height = int(max_x - min_x), int(max_y - min_y)

    results_probs = np.zeros((2, width, height), dtype=np.float32)

    for x in tqdm(range(min_x, max_x - 1, step), desc=f"Scanning polygon {index}"):
        discard_x1, discard_x2 = 0.1, 0.9
        if x == min_x:
            discard_x1 = 0.
        if x + patch_size >= max_x:
            x = max_x - patch_size
            discard_x2 = 1.
        for y in range(min_y, max_y - 1, step):
            discard_y1, discard_y2 = 0.1, 0.9
            if y + patch_size >= max_y:
                discard_y2 = 1.
                y = max_y - patch_size
            if y == min_y:
                discard_y1 = 0.

            x1 = (x - min_x) + int(patch_size * discard_x1)
            y1 = (y - min_y) + int(patch_size * discard_y1)
            x2 = (x - min_x) + int(patch_size * discard_x2)
            y2 = (y - min_y) + int(patch_size * discard_y2)

            mask_patch = mask[x1:x2, y1:y2]
            if np.sum(mask_patch) == 0:
                continue

            img = get_image_patch(dataset, x, y, patch_size, patch_size)
            if img is None:
                continue

            # Run MMSeg 1.x inference
            results_all = inference_model(model, img)
            
            # Extract prediction mask
            patch_pred = results_all.pred_sem_seg.data[0].cpu().numpy()
            
            patch_pred_cut = patch_pred[int(patch_size * discard_x1):int(patch_size * discard_x2),
                                        int(patch_size * discard_y1):int(patch_size * discard_y2)]
            results_probs[0, x1:x2, y1:y2] += (patch_pred_cut == 0)
            results_probs[1, x1:x2, y1:y2] += (patch_pred_cut == 1)

    results = np.argmax(results_probs, axis=0).astype(np.uint8)
    if mask is not None:
        results[mask == 0] = 0

    results_shp = polygons_from_binary_image(results, dataset.transform, dataset.crs, min_x=min_x, min_y=min_y)
    return results, results_shp


def predict_whole_shapefile(gpd_mask, dataset, model, patch_size, step):
    bounds = gpd_mask.total_bounds
    min_lat, min_lon, max_lat, max_lon = bounds[0], bounds[3], bounds[2], bounds[1]
    
    inv_transform = ~dataset.transform
    min_col, min_row = inv_transform * (min_lat, min_lon)
    min_col, min_row = int(np.floor(min_col)), int(np.floor(min_row))
    
    max_col, max_row = inv_transform * (max_lat, max_lon)
    max_col, max_row = int(np.ceil(max_col)), int(np.ceil(max_row))
    
    min_col = max(0, min_col)
    min_row = max(0, min_row)
    max_col = min(dataset.width, max_col)
    max_row = min(dataset.height, max_row)
    
    width = max_col - min_col
    height = max_row - min_row
    
    if width <= 0 or height <= 0:
        logger.warning("Empty intersection bounding box.")
        return None
        
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, row in gpd_mask.iterrows():
        geometry = row.geometry
        coords = np.array(geometry.exterior.coords)
        cols, rows = inv_transform * (coords[:, 0], coords[:, 1])
        pts = np.column_stack((cols, rows))
        pts = np.round(pts).astype(np.int32)
        pts = pts - np.array([min_col, min_row])
        cv2.fillPoly(mask, [pts], 1)
        
        for interior in geometry.interiors:
            coords = np.array(interior.coords)
            cols, rows = inv_transform * (coords[:, 0], coords[:, 1])
            pts = np.column_stack((cols, rows))
            pts = np.round(pts).astype(np.int32)
            pts = pts - np.array([min_col, min_row])
            cv2.fillPoly(mask, [pts], 0)
            
    results_probs = np.zeros((2, height, width), dtype=np.float32)
    
    row_steps = list(range(min_row, max_row - 1, step))
    for x in tqdm(row_steps, desc="Running inference"):
        discard_x1, discard_x2 = 0.1, 0.9
        if x == min_row:
            discard_x1 = 0.0
        if x + patch_size >= max_row:
            x = max_row - patch_size
            discard_x2 = 1.0
            
        for y in range(min_col, max_col - 1, step):
            discard_y1, discard_y2 = 0.1, 0.9
            if y + patch_size >= max_col:
                discard_y2 = 1.0
                y = max_col - patch_size
            if y == min_col:
                discard_y1 = 0.0
                
            x1 = (x - min_row) + int(patch_size * discard_x1)
            y1 = (y - min_col) + int(patch_size * discard_y1)
            x2 = (x - min_row) + int(patch_size * discard_x2)
            y2 = (y - min_col) + int(patch_size * discard_y2)
            
            mask_patch = mask[x1:x2, y1:y2]
            if np.sum(mask_patch) == 0:
                continue
                
            img = get_image_patch(dataset, x, y, patch_size, patch_size)
            if img is None:
                continue
                
            results_all = inference_model(model, img)
            patch_pred = results_all.pred_sem_seg.data[0].cpu().numpy()
            
            patch_pred_cut = patch_pred[int(patch_size * discard_x1):int(patch_size * discard_x2),
                                        int(patch_size * discard_y1):int(patch_size * discard_y2)]
                                        
            results_probs[0, x1:x2, y1:y2] += (patch_pred_cut == 0)
            results_probs[1, x1:x2, y1:y2] += (patch_pred_cut == 1)
            
    results = np.argmax(results_probs, axis=0).astype(np.uint8)
    results[mask == 0] = 0
    
    results_shp = polygons_from_binary_image(
        results, dataset.transform, dataset.crs, min_x=min_row, min_y=min_col
    )
    return results_shp


def run_inference(args):
    logger.info("Initializing model...")
    model = init_model(args.config, args.checkpoint, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.shp and args.tif:
        # Single-file mode
        shp_files = [args.shp]
        tif_paths = {args.shp: args.tif}
    else:
        # Directory mode
        shp_files = glob.glob(os.path.join(args.shp_dir, "*.shp"))
        logger.info(f"Found {len(shp_files)} shapefiles in {args.shp_dir}")
        tif_paths = {}
        for shp_path in shp_files:
            basename = os.path.splitext(os.path.basename(shp_path))[0]
            tif_name = basename
            if tif_name.endswith("_test"):
                tif_name = tif_name[:-5]
            tif_paths[shp_path] = os.path.join(args.tif_dir, f"{tif_name}.tif")

    for shp_path in shp_files:
        basename = os.path.splitext(os.path.basename(shp_path))[0]
        tif_path = tif_paths.get(shp_path)
        if not tif_path or not os.path.exists(tif_path):
            logger.warning(f"No matching TIFF found for {shp_path} at {tif_path}. Skipping.")
            continue

        out_shp_path = os.path.join(args.out_dir, f"{basename}_pred.shp")
        logger.info(f"Processing shapefile: {basename}")
        logger.info(f"Using TIFF: {tif_path}")

        gpd_mask = gpd.read_file(shp_path)
        gpd_mask = gpd_mask.explode(ignore_index=True)
        gpd_mask = gpd_mask[~gpd_mask.geometry.is_empty & gpd_mask.geometry.notna()]
        gpd_mask = gpd_mask.reset_index(drop=True)
        with rasterio.open(tif_path) as dataset:
            gpd_mask = gpd_mask.to_crs(dataset.crs)
            results_shp = predict_whole_shapefile(
                gpd_mask, dataset, model, args.patch_size, args.step
            )

            if results_shp is not None and not results_shp.empty:
                results_shp.to_file(out_shp_path)
                logger.info(f"Saved prediction shapefile to {out_shp_path}")
            else:
                logger.warning(f"No predictions found inside any polygons of {basename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding-window MMSeg shapefile inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--shp", type=str, default=None, help="Path to a single shapefile (overrides shp-dir)")
    parser.add_argument("--tif", type=str, default=None, help="Path to a single TIFF file (overrides tif-dir)")
    parser.add_argument("--shp-dir", type=str, default="pantanal/queimadas/data/shapes/test", help="Shapes directory")
    parser.add_argument("--tif-dir", type=str, default="pantanal/queimadas/original_data/tifs", help="TIFF directory")
    parser.add_argument("--out-dir", type=str, default="pantanal/queimadas/predictions", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size")
    parser.add_argument("--step", type=int, default=128, help="Stride step size")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    run_inference(args)

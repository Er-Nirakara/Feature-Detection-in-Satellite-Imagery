import os
import cv2
import rasterio
import numpy as np
import pandas as pd
import shapely.wkt
import shapely.affinity
from tqdm import tqdm
import gc

import config

def get_grid_sizes():
    """Reads grid sizes to scale WKT coordinates to image pixels."""
    grid_df = pd.read_csv(config.GRID_SIZES_PATH)
    grid_sizes = {}
    for _, row in grid_df.iterrows():
        grid_sizes[row['Unnamed: 0']] = (row['Xmax'], row['Ymin'])
    return grid_sizes

def read_and_align_multispectral(image_id):
    """Aligns RGB and NIR bands, standardizes to [0, 1] floats."""
    # GDAL bug fix: Force forward slashes for Windows paths with spaces
    rgb_path = os.path.join(config.RGB_DIR, f"{image_id}.tif").replace('\\', '/')
    mband_path = os.path.join(config.MBAND_DIR, f"{image_id}_M.tif").replace('\\', '/')
    
    # 1. Read RGB
    with rasterio.open(rgb_path) as src_rgb:
        # rasterio reads as (channels, height, width). Transpose to (H, W, C)
        rgb_img = src_rgb.read().transpose(1, 2, 0)
        h, w, _ = rgb_img.shape
        
    # 2. Read M-Band & extract NIR (DSTL M-Band channel 7 is NIR1)
    with rasterio.open(mband_path) as src_m:
        m_img = src_m.read()
        nir_band = m_img[6] # 0-indexed, so 6 is the 7th channel
        
    # 3. Geometric Alignment (Upscale NIR to match RGB dimensions)
    nir_aligned = cv2.resize(nir_band, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 4. Stack into 4-Channel Array (RGB + NIR)
    # Add an axis to NIR to make it (H, W, 1) before concatenating
    img_4c = np.concatenate([rgb_img, nir_aligned[:, :, np.newaxis]], axis=-1)
    
    # 5. Radiometric Normalization (Min-Max scaling to 0.0 - 1.0)
    img_4c = img_4c.astype(np.float32)
    for c in range(4):
        ch_min = img_4c[:, :, c].min()
        ch_max = img_4c[:, :, c].max()
        if ch_max > ch_min:
            img_4c[:, :, c] = (img_4c[:, :, c] - ch_min) / (ch_max - ch_min)
            
    return img_4c, h, w

def generate_target_masks(image_id, h, w, xmax, ymin, wkt_df):
    """Converts WKT polygons into a 3-channel binary mask (Buildings, Roads, Veg)."""
    
    # Create separate 2D arrays to prevent OpenCV memory layout errors
    mask_c0 = np.zeros((h, w), dtype=np.float32) # Buildings
    mask_c1 = np.zeros((h, w), dtype=np.float32) # Roads
    mask_c2 = np.zeros((h, w), dtype=np.float32) # Vegetation
    masks = [mask_c0, mask_c1, mask_c2]
    
    # DSTL Scale factors
    w_scaler = w / xmax
    h_scaler = h / ymin
    
    class_mapping = {1: 0, 3: 1, 5: 2, 6: 2}
    
    img_polygons = wkt_df[wkt_df['ImageId'] == image_id]
    
    for _, row in img_polygons.iterrows():
        class_id = row['ClassType']
        if class_id not in class_mapping:
            continue # Skip excluded classes
            
        channel_idx = class_mapping[class_id]
        multipolygon = shapely.wkt.loads(row['MultipolygonWKT'])
        
        if multipolygon.is_empty:
            continue
            
        # Scale mathematical polygons to pixel space
        scaled_poly = shapely.affinity.scale(multipolygon, xfact=w_scaler, yfact=h_scaler, origin=(0,0,0))
        
        # Function to extract coordinates and draw
        def draw_poly(geom):
            if geom.geom_type == 'Polygon':
                # OpenCV requires coordinates as integers
                ext_coords = np.array(geom.exterior.coords).round().astype(np.int32)
                # Draw on the specific 2D channel
                cv2.fillPoly(masks[channel_idx], [ext_coords], 1.0)
            elif geom.geom_type == 'MultiPolygon':
                for p in geom.geoms:
                    draw_poly(p)
                    
        draw_poly(scaled_poly)
        
    # Stack the 3 separate 2D channels back into a single (H, W, 3) image array
    return np.stack(masks, axis=-1)

def create_and_save_tiles(image_id, img_array, mask_array):
    """Sliding window algorithm to crop and save 128x128 patches."""
    h, w, _ = img_array.shape
    patch_size = config.PATCH_SIZE
    stride = config.STRIDE
    
    tiles_saved = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = img_array[y:y+patch_size, x:x+patch_size, :]
            mask_patch = mask_array[y:y+patch_size, x:x+patch_size, :]
            
            # To prevent model from learning on completely empty background tiles,
            # we can skip tiles that are 100% empty/black, but for strict map 
            # continuity we will save all of them.
            
            img_filename = os.path.join(config.CACHE_DIR, f"{image_id}_{y}_{x}_img.npy")
            mask_filename = os.path.join(config.CACHE_DIR, f"{image_id}_{y}_{x}_mask.npy")
            
            np.save(img_filename, img_patch)
            np.save(mask_filename, mask_patch)
            tiles_saved += 1
            
    print(f"[{image_id}] Generated {tiles_saved} micro-tensors.")

def main():
    print("--- Phase 1 & 2: Multispectral Alignment & WKT Parsing ---")
    grid_sizes = get_grid_sizes()
    wkt_df = pd.read_csv(config.WKT_CSV_PATH)
    
    for img_id in config.IMAGE_IDS:
        print(f"\nProcessing Satellite Swath: {img_id}")
        
        # 1. Read & Align Multispectral Bands
        print("  -> Aligning RGB and NIR bands...")
        img_4c, h, w = read_and_align_multispectral(img_id)
        
        # 2. Parse WKT into Raster Masks
        print("  -> Converting WKT to Pixel Masks...")
        xmax, ymin = grid_sizes[img_id]
        mask = generate_target_masks(img_id, h, w, xmax, ymin, wkt_df)
        
        # 3. Spatial Fragmentation (Tiling Engine)
        print("  -> Running Sliding Window Tiling Engine...")
        create_and_save_tiles(img_id, img_4c, mask)
        
        # Clear RAM immediately to prevent System Freeze (Hardware constraint bypass)
        del img_4c, mask
        gc.collect()
        
    print("\n--- Data Pipeline Completed Successfully! ---")
    print(f"Check your SSD cache folder: {config.CACHE_DIR}")

if __name__ == "__main__":
    main()

    
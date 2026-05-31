import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
RGB_DIR = os.path.join(DATA_DIR, 'three_band')
MBAND_DIR = os.path.join(DATA_DIR, 'sixteen_band')
WKT_CSV_PATH = os.path.join(DATA_DIR, 'train_wkt_v4.csv')
GRID_SIZES_PATH = os.path.join(DATA_DIR, 'grid_sizes.csv')

# SSD Lazy Loading Cache Directory
CACHE_DIR = os.path.join(BASE_DIR, 'cache', 'tiles')
os.makedirs(CACHE_DIR, exist_ok=True)

# --- PoC DATASET SUBSET ---
# Using 3-5 images as defined in the PRD to save your 70GB SSD
IMAGE_IDS = ['6120_2_2', '6120_2_0', '6100_1_3'] 

# --- HARDWARE & TILING CONSTRAINTS ---
PATCH_SIZE = 128
OVERLAP = 28
STRIDE = PATCH_SIZE - OVERLAP

# --- DEEP LEARNING HYPERPARAMETERS ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20
IN_CHANNELS = 4 # RGB + NIR
OUT_CLASSES = 3 # Buildings, Roads, Vegetation (Trees + Crops)
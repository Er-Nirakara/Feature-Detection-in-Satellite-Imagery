# Feature Detection in Satellite Imagery using Deep Learning 🌍🛰️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Rasterio](https://img.shields.io/badge/Rasterio-Geospatial-lightgrey.svg)

## Project Overview
This project is an automated Semantic Segmentation pipeline designed to identify physical features (Buildings, Roads, and Vegetation) in high-resolution, multi-sensor satellite imagery. It was developed as part of a Work-Based Learning (WBL) Internship Project at **NIELIT, Bhubaneswar**.

The primary engineering achievement of this project is the construction of a highly optimized, resource-efficient data pipeline capable of training a Convolutional Neural Network (CNN) under severe local hardware constraints (**8GB RAM, GTX 1650 4GB VRAM**) without triggering Out-of-Memory (OOM) crashes.

## Core Engineering Solutions
1. **Multispectral Alignment:** Extracted and aligned the Near-Infrared (NIR) band from lower-resolution M-Band sensors with the high-resolution RGB image using cubic interpolation.
2. **Geospatial Processing:** Converted complex Well-Known Text (WKT) multi-polygons into continuous pixel-based 2D binary raster masks using `Shapely` and `OpenCV`.
3. **Hardware Optimization (Tiling Engine):** Designed a sliding-window algorithm to crop massive 3000x3000 satellite swaths into 128x128 micro-tensors.
4. **SSD-Based Lazy Loading:** Bypassed the 8GB RAM bottleneck by streaming micro-tensors directly from the SSD to the GPU during the PyTorch training loop.
5. **Lightweight U-Net:** Downscaled a standard U-Net architecture from ~31 million parameters to exactly **1,944,227 parameters**, allowing active training batches to fit inside 4GB of VRAM.

## Repository Structure
```text
├── dataset_prep.py     # Parses WKT, aligns multispectral bands, runs Tiling Engine
├── train.py            # Custom Lazy DataLoader, BCE+Dice Loss, U-Net training loop
├── model.py            # PyTorch implementation of the Lightweight U-Net
├── visualize.py        # Generates Matplotlib comparisons of predictions
├── config.py           # Global hyperparameter and path configurations
└── README.md

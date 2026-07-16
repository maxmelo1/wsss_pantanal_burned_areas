# An Uncertainty-Aware Deep Learning Approach for Rapid Burned Area Mapping using High-Resolution PlanetScope Multispectral Imagery

This is the official repository for **Weakly Supervised Semantic Segmentation (WSSS) for Burned Area Mapping in the Brazilian Pantanal**.

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Document-Paper-blue.svg?style=flat)]()

---

## Overview

Wildfire mapping in remote and ecologically sensitive regions like the Brazilian Pantanal faces critical challenges due to the high costs associated with collecting pixel-level annotations required for fully supervised models. 

This project implements an **uncertainty-aware weakly supervised pipeline** to map burned areas using high-resolution, four-band (**RGB-NIR**) PlanetScope satellite imagery:

*   **Two-Stage Pipeline:** First, we generate high-quality pseudo-labels from cheap image-level annotations using adapted WSSS methods (**SEAM** and **Puzzle-CAM**). Second, we train a robust **SegFormer** segmentation model on these generated pseudo-labels.
*   **Spectral Adaptation:** Both CAM-based architectures were adapted to accept 4-channel input data to leverage the Near-Infrared (NIR) band, which is crucial for distinguishing burned areas from background soil and vegetation.
*   **Key Finding:** Puzzle-CAM combined with a strong **ResNeSt-101** backbone produces outstanding pseudo-labels, yielding segmentation performance that closely approaches fully supervised methods while drastically reducing manual labeling efforts.

---

## Prerequisites & Hardware

*   **Hardware:** An NVIDIA GPU with **at least 12 GB of VRAM** (e.g., RTX 3060/4070 or better) is highly recommended for training.
*   **Software Requirements:** This project shares the same environment configurations as the original Puzzle-CAM framework.
    *   Linux (Ubuntu recommended)
    *   CUDA-compatible PyTorch environment

---

## Getting Started

### 1. Installation
Clone this repository and set up the environment based on the core [Puzzle-CAM Repository](https://github.com/shjo-april/PuzzleCAM/):

```bash
git clone [https://github.com/maxmelo1/wsss_pantanal_burned_areas](https://github.com/maxmelo1/wsss_pantanal_burned_areas)
cd wsss_pantanal_burned_areas
# Follow the environment setup instructions from Puzzle-CAM
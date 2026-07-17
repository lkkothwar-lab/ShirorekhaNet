# ShirorekhaNet: Multi‑Scale Attention Network with Cosine‑LR PID Curriculum

This repository contains the final implementation of **ShirorekhaNet** for text‑line‑level segmentation of Sanskrit manuscripts.  
The training strategy uses a **CosineLR‑PID‑Curriculum** that jointly schedules the learning rate and dynamically balances Dice, BCE, and perceptual loss weights.

## New in this version

- **ShirorekhaNet** architecture: U‑Net style encoder‑decoder with `MultiScaleInception` blocks and channel attention.
- **CosineLR‑PID‑Curriculum**: Simultaneous cosine learning‑rate decay and PID‑controlled loss‑weight adjustment.
- **Simplified data pipeline**: Inputs are 4‑channel (RGB + adaptive‑thresholded binary image). No external augmentation library required.
- **Comprehensive metrics**: IoU, Dice, F1, precision, recall, detection rate, and custom Shirorekha accuracy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19152726.svg)](https://doi.org/10.5281/zenodo.19152726)

## Repository structure
├── dataset.py # ShirorekhaDataset (4‑channel input)

├── model.py # ShirorekhaNet architecture

├── loss.py # SanskritLoss + CosineLR_PID_Curriculum

├── utils.py # Metrics and visualisation

├── train.py # Training script

├── README.md

└── dataset/ # (place your data here)


## Dataset: Shirorekha Annotations for Devanagari Manuscripts (125 Folios)

### Dataset Overview
This repository utilizes a custom dataset providing pixel-wise line masks (Shirorekha annotations) for 125 selected scanned Devanagari manuscript folios. These annotations are specifically designed to support line-level segmentation and historical manuscript analysis.

The original images are sourced from the PE-OCR Sanskrit benchmark dataset introduced by Maheshwari et al. (2022).  
**Original Source:** [https://github.com/ayushbits/pe-ocr-sanskrit](https://github.com/ayushbits/pe-ocr-sanskrit) 

### Annotation Details
* **Target:** Shirorekha (header line) of Devanagari characters.
* **Format:** Binary PNG masks.
* **Pixel Values:** 255 (Foreground/Shirorekha), 0 (Background).
* **Process:** Each folio was manually annotated by two persons.

### Data Split
The 125 folios are partitioned as follows:
* **Training:** 85 folios
* **Validation:** 15 folios
* **Test:** 25 folios

*Note: While data augmentation is recommended for training (handled dynamically via Albumentations in our code), the provided dataset contains the original, non-augmented masks.*

<details>
<summary><b>Click to view the full 125-Folio Filename Mapping Table</b></summary>

To maintain consistency, the original images from the PE-OCR source were renamed. The table below provides the mapping between the original source directory/number and the renamed filename in this annotated dataset.

| Sr. | Source DS | Page Range | Orig. No | Renamed Name |
|---|---|---|---|---|
| 1 | BHSv1S1 | Page-001-102 | 6 | img-001 |
| 2 | BHSv1S1 | Page-001-102 | 8 | img-002 |
| 3 | BHSv1S1 | Page-001-102 | 9 | img-003 |
| 4 | BHSv1S1 | Page-001-102 | 10 | img-004 |
| 5 | BHSv1S1 | Page-001-102 | 21 | img-005 |
| 6 | BHSv1S1 | Page-001-102 | 29 | img-006 |
| 7 | BHSv1S1 | Page-001-102 | 32 | img-007 |
| 8 | BHSv1S1 | Page-001-102 | 37 | img-008 |
| 9 | BHSv1S1 | Page-001-102 | 42 | img-009 |
| 10 | BHSv1S1 | Page-001-102 | 43 | img-010 |
| 11 | BHSv1S1 | Page-001-102 | 44 | img-011 |
| 12 | BHSv1S1 | Page-001-102 | 47 | img-012 |
| 13 | BHSv1S1 | Page-001-102 | 49 | img-013 |
| 14 | BHSv1S1 | Page-001-102 | 51 | img-014 |
| 15 | BHSv1S1 | Page-001-102 | 52 | img-015 |
| 16 | BHSv1S1 | Page-001-102 | 53 | img-016 |
| 17 | BHSv1S1 | Page-001-102 | 57 | img-017 |
| 18 | BHSv1S1 | Page-001-102 | 65 | img-018 |
| 19 | BHSv1S1 | Page-001-102 | 70 | img-019 |
| 20 | BHSv1S1 | Page-001-102 | 76 | img-020 |
| 21 | BHSv1S1 | Page-001-102 | 80 | img-021 |
| 22 | BHSv1S1 | Page-001-102 | 85 | img-022 |
| 23 | BHSv1S1 | Page-001-102 | 88 | img-023 |
| 24 | BHSv1S1 | Page-001-102 | 95 | img-024 |
| 25 | BHSv1S1 | Page-001-102 | 100 | img-025 |
| 26 | BHSv1S4 | Page-309-326 | 310 | img-026 |
| 27 | BHSv1S4 | Page-309-326 | 311 | img-027 |
| 28 | BHSv1S4 | Page-309-326 | 314 | img-028 |
| 29 | BHSv1S4 | Page-309-326 | 315 | img-029 |
| 30 | BHSv1S4 | Page-309-326 | 317 | img-030 |
| 31 | BHSv1S4 | Page-309-326 | 320 | img-031 |
| 32 | BHSv1S4 | Page-309-326 | 323 | img-032 |
| 33 | BHSv1S4 | Page-309-326 | 324 | img-033 |
| 34 | BHSv1S4 | Page-309-326 | 325 | img-034 |
| 35 | BHSv1S4 | Page-309-326 | 326 | img-035 |
| 36 | GG1 | page-002-100 | 3 | img-036 |
| 37 | GG1 | page-002-100 | 4 | img-037 |
| 38 | GG1 | page-002-100 | 5 | img-038 |
| 39 | GG1 | page-002-100 | 6 | img-039 |
| 40 | GG1 | page-002-100 | 13 | img-040 |
| 41 | GG1 | page-002-100 | 14 | img-041 |
| 42 | GG1 | page-002-100 | 15 | img-042 |
| 43 | GG1 | page-002-100 | 17 | img-043 |
| 44 | GG1 | page-002-100 | 19 | img-044 |
| 45 | GG1 | page-002-100 | 24 | img-045 |
| 46 | GG1 | page-002-100 | 26 | img-046 |
| 47 | GG1 | page-002-100 | 28 | img-047 |
| 48 | GG1 | page-002-100 | 62 | img-048 |
| 49 | GG1 | page-002-100 | 65 | img-049 |
| 50 | GG1 | page-002-100 | 74 | img-050 |
| 51 | GG1 | page-002-100 | 78 | img-051 |
| 52 | GG1 | page-002-100 | 79 | img-052 |
| 53 | GG1 | page-002-100 | 80 | img-053 |
| 54 | GG1 | page-002-100 | 81 | img-054 |
| 55 | GG1 | page-002-100 | 82 | img-055 |
| 56 | GG1 | page-002-100 | 83 | img-056 |
| 57 | GG1 | page-002-100 | 84 | img-057 |
| 58 | GG1 | page-002-100 | 85 | img-058 |
| 59 | GG1 | page-002-100 | 92 | img-059 |
| 60 | GG1 | page-002-100 | 100 | img-060 |
| 61 | GG2 | page-101-200 | 101 | img-061 |
| 62 | GG2 | page-101-200 | 102 | img-062 |
| 63 | GG2 | page-101-200 | 103 | img-063 |
| 64 | GG2 | page-101-200 | 104 | img-064 |
| 65 | GG2 | page-101-200 | 106 | img-065 |
| 66 | GG2 | page-101-200 | 108 | img-066 |
| 67 | GG2 | page-101-200 | 110 | img-067 |
| 68 | GG2 | page-101-200 | 113 | img-068 |
| 69 | GG2 | page-101-200 | 115 | img-069 |
| 70 | GG2 | page-101-200 | 121 | img-070 |
| 71 | GG2 | page-101-200 | 125 | img-071 |
| 72 | GG2 | page-101-200 | 127 | img-072 |
| 73 | GG2 | page-101-200 | 128 | img-073 |
| 74 | GG2 | page-101-200 | 129 | img-074 |
| 75 | GG2 | page-101-200 | 131 | img-075 |
| 76 | GG2 | page-101-200 | 134 | img-076 |
| 77 | GG2 | page-101-200 | 139 | img-077 |
| 78 | GG2 | page-101-200 | 148 | img-078 |
| 79 | GG2 | page-101-200 | 156 | img-079 |
| 80 | GG2 | page-101-200 | 170 | img-080 |
| 81 | GG2 | page-101-200 | 172 | img-081 |
| 82 | GG2 | page-101-200 | 189 | img-082 |
| 83 | GG2 | page-101-200 | 195 | img-083 |
| 84 | GG2 | page-101-200 | 196 | img-084 |
| 85 | GG2 | page-101-200 | 199 | img-085 |
| 86 | GG3 | page-201-300 | 201 | img-086 |
| 87 | GG3 | page-201-300 | 204 | img-087 |
| 88 | GG3 | page-201-300 | 205 | img-088 |
| 89 | GG3 | page-201-300 | 206 | img-089 |
| 90 | GG3 | page-201-300 | 214 | img-090 |
| 91 | GG3 | page-201-300 | 217 | img-091 |
| 92 | GG3 | page-201-300 | 220 | img-092 |
| 93 | GG3 | page-201-300 | 222 | img-093 |
| 94 | GG3 | page-201-300 | 224 | img-094 |
| 95 | GG3 | page-201-300 | 234 | img-095 |
| 96 | GG3 | page-201-300 | 240 | img-096 |
| 97 | GG3 | page-201-300 | 251 | img-097 |
| 98 | GG3 | page-201-300 | 254 | img-098 |
| 99 | GG3 | page-201-300 | 267 | img-099 |
| 100 | GG3 | page-201-300 | 269 | img-100 |
| 101 | GG3 | page-201-300 | 280 | img-101 |
| 102 | GG3 | page-201-300 | 289 | img-102 |
| 103 | GG3 | page-201-300 | 296 | img-103 |
| 104 | GG3 | page-201-300 | 298 | img-104 |
| 105 | GG3 | page-201-300 | 300 | img-105 |
| 106 | GOSv1S2 | page-101-150 | 101 | img-106 |
| 107 | GOSv1S2 | page-101-150 | 102 | img-107 |
| 108 | GOSv1S2 | page-101-150 | 106 | img-108 |
| 109 | GOSv1S2 | page-101-150 | 107 | img-109 |
| 110 | GOSv1S2 | page-101-150 | 108 | img-110 |
| 111 | GOSv1S2 | page-101-150 | 116 | img-111 |
| 112 | GOSv1S2 | page-101-150 | 118 | img-112 |
| 113 | GOSv1S2 | page-101-150 | 142 | img-113 |
| 114 | GOSv1S2 | page-101-150 | 143 | img-114 |
| 115 | GOSv1S2 | page-101-150 | 144 | img-115 |
| 116 | GOSv1S3 | page-151-195 | 151 | img-116 |
| 117 | GOSv1S3 | page-151-195 | 154 | img-117 |
| 118 | GOSv1S3 | page-151-195 | 158 | img-118 |
| 119 | GOSv1S3 | page-151-195 | 166 | img-119 |
| 120 | GOSv1S3 | page-151-195 | 168 | img-120 |
| 121 | GOSv1S3 | page-151-195 | 170 | img-121 |
| 122 | GOSv1S3 | page-151-195 | 172 | img-122 |
| 123 | GOSv1S3 | page-151-195 | 175 | img-123 |
| 124 | GOSv1S3 | page-151-195 | 185 | img-124 |
| 125 | GOSv1S3 | page-151-195 | 194 | img-125 |

</details>

# Prepare your dataset
Organise your images and binary masks into train_data/Trn_img, train_data/Trn_lmask, val_data/Vld_img, val_data/Vld_lmask (see train.py for paths).

# Train the model
python train.py
By default, the script will save checkpoints, logs, and epoch visualisations inside output/.

# Resume training
The script automatically detects the latest checkpoint and resumes from it.


---

**Note:**  
- In `train.py`, you need to adjust `BASE_PATH` and folder names to match your local setup.  
- The script expects the data already unzipped; you can add the unzip step from the Colab if you prefer.  
- All files are self‑contained and match the behaviour of your latest Colab implementation.  

Once you replace the old files with these, your repository will reflect the final proposed method. Let me know if you need any further adjustments.

## Sample Visualization

| Sample 1 with long inter word distance | Sample 2 with dense lines|
|----------|--------------|---------------------------|
| ![Sample 1 with long inter word distance](visual_long.jpg) | ![Sample 2 with dense lines](visual_dense.jpg) |

### Install dependencies
```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm


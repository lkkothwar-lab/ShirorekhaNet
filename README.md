# ShirorekhaNet: A Multi-Scale Attention Network for Text Line-Level Segmentation of Sanskrit Manuscripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
*(space for link/badge here once published)*

Official PyTorch implementation of **ShirorekhaNet**, introduced in our paper: *ShirorekhaNet: A Multi-Scale Attention Network for Text Line-Level Segmentation of Sanskrit Manuscripts*.

## Abstract
Text line segmentation in historical Sanskrit manuscripts remains challenging due to physical deterioration, irregular layouts, and script-specific features such as the Shirorekha (header line) and densely packed conjuncts. For pixel-accurate text line-level segmentation, we introduce ShirorekhaNet, a residual encoder-decoder network. A four-channel input (RGB + binarized image) is accepted by the architecture, which combines three essential elements: residual connections with learnable gating, multi-scale Inception (MSI) modules (1×1, 3×3, 5×5) enhanced by channel attention, and a two-stage training strategy combining fixed warmup with PID-controlled adaptation. 

In just eight epochs, this hybrid approach reaches a validation IoU of 0.9016. On the test set, ShirorekhaNet achieves:
* **Precision:** 0.9473
* **Recall:** 0.9495
* **IoU:** 0.9019
* **Dice Coefficient:** 0.9482
* **Shirorekha Accuracy:** 96.40%

## Repository Structure

The codebase is organized into modular Python scripts for easy readability and integration:

* `dataset.py`: Contains the `SanskritLineDataset` class and Albumentations augmentations.
* `model.py`: Defines the `ShirorekhaNet` architecture, including the `DevanagariInception` and `ChannelAttention` modules.
* `loss.py`: Houses the custom `SanskritLoss` function and the `HyperAggressiveAdaptiveMetaTuner` (PID controller).
* `utils.py`: Includes metric calculations (IoU, Shirorekha accuracy), the `ModelSaver`, and the `Visualizer`.
* `train.py`: The main execution script to initialize the dataset, model, and begin the training loop.
* `/dataset`: Directory for the training and validation data (not tracked by Git).
* `/checkpoints`: Directory where model weights are saved during training.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision opencv-python numpy albumentations matplotlib scikit-learn tqdm

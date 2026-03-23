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

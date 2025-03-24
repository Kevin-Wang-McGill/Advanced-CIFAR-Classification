# Advanced-CIFAR-Classification

An advanced Convolutional Neural Network (CNN) project for image classification on **CIFAR-10** and **CIFAR-100**. This repository explores deeper network architectures, regularization techniques, and interpretability methods (including **Grad-CAM**) to provide insights into how the model makes decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training on CIFAR-10](#training-on-cifar-10)
  - [Fine-Tuning on CIFAR-100](#fine-tuning-on-cifar-100)
  - [Grad-CAM Visualization](#grad-cam-visualization)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This project demonstrates:

1. **Data Preprocessing & Augmentation**: Normalization, random cropping, horizontal flipping, and color jittering.  
2. **CNN Architecture**: Incorporates advanced components like Residual Blocks, Batch Normalization, and Dropout.  
3. **Training & Optimization**: Uses learning rate scheduling (e.g., `ReduceLROnPlateau`) and either `Adam` or `SGD + Momentum`.  
4. **Evaluation & Analysis**: Measures accuracy, precision, recall, F1-score, and confusion matrices on test sets.  
5. **Visualization & Interpretability**: Generates Grad-CAM heatmaps to highlight regions influencing model decisions.

---

## Features

- **CIFAR-10 & CIFAR-100** support
- **Residual Connections** for deeper, more robust networks
- **Dropout & Batch Normalization** to reduce overfitting and improve training stability
- **Learning Rate Scheduling** to fine-tune convergence
- **Grad-CAM** for model interpretability
- **Optional Feature Map Visualization** to see how the network learns

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/Advanced-CIFAR-Classification.git
   cd Advanced-CIFAR-Classification
# Advanced CIFAR Classification

## Installation

Install dependencies (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```

Or manually install packages such as:
- torch / torchvision
- numpy
- matplotlib
- opencv-python
- scikit-learn

## Environment Requirements
- Python 3.7+
- GPU with CUDA (optional but recommended)

## Data Preparation
By default, the code uses PyTorch's `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100`. The datasets will be downloaded automatically to the `./data` directory if not found.

You can customize the path or the data loading by editing the data-loading scripts (e.g., `data_loader.py` if provided).

## Usage

### Training on CIFAR-10
Train:
```bash
python train_cifar10.py
```

Adjust hyperparameters (batch size, epochs, learning rate, etc.) in the script if needed.

Monitor:
- Training and validation metrics (accuracy, loss) are printed per epoch.
- The best model checkpoint is saved as `best_model.pth`.

### Fine-Tuning on CIFAR-100
Load CIFAR-10 weights and filter out the final classification layer:
```python
model_100 = CNN_Model(num_classes=100)
pretrained_dict = torch.load("best_model.pth")
filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc.")}
model_100.load_state_dict(filtered_dict, strict=False)
```

Fine-tune:
```bash
python train_cifar100.py
```

Compare results on CIFAR-100 with and without fine-tuning.

### Grad-CAM Visualization
Use the `grad_cam.py` script (or an equivalent notebook) to generate Grad-CAM heatmaps:
```bash
python grad_cam.py
```

This script loads the trained model, selects a few images from the test set, and displays the original image overlaid with a Grad-CAM heatmap.

## Results

| Model  | Dataset   | Test Accuracy | Precision | Recall | F1 Score |
|--------|-----------|---------------|-----------|--------|----------|
| ResNet | CIFAR-10  | ~90%          | ~90%      | ~90%   | ~90%     |
| ResNet | CIFAR-100 | ~65%          | ~65%      | ~65%   | ~65%     |

Sample Grad-CAM output: [Add image here]

## Project Structure
```
Advanced-CIFAR-Classification/
│
├─ data/            # Default location for CIFAR-10 / CIFAR-100
├─ models/          # Model definitions (e.g., CNN_Model with residual blocks)
├─ scripts/         # Training scripts, Grad-CAM script
├─ utils/           # Helper functions (logging, transforms, etc.)
├─ train_cifar10.py # Entry point for CIFAR-10 training
├─ train_cifar100.py # Entry point for CIFAR-100 fine-tuning
├─ grad_cam.py      # Grad-CAM visualization script
├─ README.md
├─ requirements.txt
└─ LICENSE
```
Adjust to your actual file layout.

## License
This project is licensed under the MIT License. Feel free to use and modify for your own purposes.

Enjoy exploring advanced CNN techniques with CIFAR-10 and CIFAR-100!

MIT License

Copyright (c) [2025] [Kevin]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

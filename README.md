# MSCA-RTDETR  

MSCA-RTDETR: Multi-Scale Context-Aligned Real-Time DETR for Fabric Defect Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  





## 🚀 Features

- **Real-time Detection**
- **Multi-Scale Processing**
- **Distributed Training**: Support for multi-GPU training
- **Comprehensive Evaluation**: Built-in metrics and visualization tools

## 📋 Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Training](#training)
- [Testing](#testing)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.9+

### Environment Setup

#### 1. Mamba Module Compilation

```bash
cd ultralytics/nn/extra_modules/mamba
pip install causal-conv1d
pip install mamba-ssm
python setup.py install
```

#### 2. Basic Dependencies

```bash
pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets psutil
```

#### 3. MMEngine & MMCV Installation

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.2.0"
```

## 🏋️ Training

### Single GPU Training

```bash
python train.py
```

### Multi-GPU Training

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py
```

> **Note**: Replace `2` with the number of GPUs you want to use.

### Training Parameters

- `--nproc_per_node`: Number of processes per node (typically equals number of GPUs)
- Additional parameters can be configured in the training script

## 🧪 Testing

Run inference on your dataset:

```bash
python detect.py
```

## 📊 Dataset

### Dataset Information

For complete dataset information and download links, please refer to:
```
dataset/datasets_download.txt
```

### Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── datasets_download.txt
```

### Contact for Dataset

For complete dataset information and access, please contact the author.



## 🛠️ Project Structure

```
MCAS-RTDETR/
├── ultralytics/           # Core detection framework
│   └── nn/
│       └── extra_modules/
│           └── mamba/     # Mamba module
├── dataset/               # Dataset directory
├── train.py              # Training script
├── detect.py             # Detection script
├── val.py                # Validation script
├── export.py             # Model export script
├── requirements.txt      # Dependencies
└── README.md            # This file
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 📧 Contact

For questions about the dataset or project, please contact the author.




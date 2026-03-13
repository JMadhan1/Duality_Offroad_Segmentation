#!/usr/bin/env bash
# ============================================================
#  Duality AI Offroad Segmentation — Linux/Mac Environment Setup
# ============================================================
set -e

echo "Creating conda environment 'EDU' with Python 3.10..."
conda create -n EDU python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate EDU

echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing segmentation libraries..."
pip install segmentation-models-pytorch==0.3.3
pip install transformers timm

echo "Installing data / augmentation libraries..."
pip install albumentations opencv-python-headless Pillow numpy

echo "Installing training utilities..."
pip install tensorboard tqdm pandas scikit-learn matplotlib seaborn PyYAML

echo "Installing TTA and extras..."
pip install ttach

echo "============================================================"
echo "Setup complete!  Run:  conda activate EDU"
echo "Then:            cd project && python train.py"
echo "============================================================"

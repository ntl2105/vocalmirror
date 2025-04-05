#!/bin/bash

set -e  # exit on error

echo "🚀 Starting environment setup for WhisperX..."

# Update and install dependencies
sudo apt update
sudo apt install -y git wget ffmpeg build-essential python3-pip python3-venv

# Install Miniconda (if not already installed)
if ! command -v conda &> /dev/null; then
    echo "🟢 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Create a fresh environment
echo "🟢 Creating conda environment 'whisperx_gpu'..."
conda create -y -n whisperx_gpu python=3.10
source activate whisperx_gpu

# Install PyTorch with CUDA support (adjust version as needed!)
echo "🟢 Installing PyTorch with CUDA 12.1 support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install WhisperX and dependencies
echo "🟢 Installing WhisperX..."
pip install git+https://github.com/m-bain/whisperx.git

# Install other optional tools
echo "🟢 Installing ffmpeg, tqdm, and other tools..."
pip install ffmpeg-python tqdm rich

# Check CUDA availability
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"

echo "🎉 Environment setup completed! To use:"
echo "source activate whisperx_gpu"
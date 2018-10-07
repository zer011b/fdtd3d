#!/bin/bash

set -ex

CUDA_REPO_PKG=cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
rm ${CUDA_REPO_PKG}

echo 'export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
' >> ~/.bashrc

# Let terminal know of the changes to the .bashrc file
source ~/.bashrc

sudo apt-get update

# y flag just says yes to all prompts
sudo apt-get install -y --allow-unauthenticated cuda

# Check if installation is successful by running the next line
nvcc -V

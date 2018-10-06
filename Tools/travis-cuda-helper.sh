#!/bin/bash

CUDA_REPO_PKG=cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
rm ${CUDA_REPO_PKG}
sudo apt-get install cuda

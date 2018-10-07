#!/bin/bash

set -ex

sudo curl -sSL https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.tar.gz | sudo tar -xzC /opt
export PATH=/opt/cmake-3.12.3-Linux-x86_64/bin:$PATH
export LD_LIBRARY_PATH=/opt/cmake-3.12.3-Linux-x86_64/lib:$LD_LIBRARY_PATH
export HOME_DIR=`pwd`
export BUILD_DIR="${HOME_DIR}/Build"
mkdir ${BUILD_DIR}

#!/bin/bash

set -ex

export HOME_DIR=`pwd`
export BUILD_DIR="${HOME_DIR}/Build"
mkdir ${BUILD_DIR}

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-6 g++-6

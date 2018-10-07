#!/bin/bash

set -ex

git clone https://gitlab.kitware.com/cmake/cmake.git ${PROJECT_SOURCE_DIR}/Third-party/cmake
cd ${PROJECT_SOURCE_DIR}/Third-party/cmake
./bootstrap
make

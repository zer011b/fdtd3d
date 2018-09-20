#!/bin/bash

set -e

git clone https://gitlab.kitware.com/cmake/cmake.git ${PROJECT_SOURCE_DIR}/Third-party/cmake
cd ${PROJECT_SOURCE_DIR}/Third-party/cmake
./bootstrap
make
#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

USE_CUDA=$3

CUDA_MODE=""
if [[ "$USE_CUDA" -eq "1" ]]; then
  CUDA_MODE="-DSOLVER_DIM_MODES=DIM3 -DCUDA_ENABLED=ON -DCUDA_ARCH_SM_TYPE=sm_35"
fi

TEST_DIR=$(dirname $(readlink -f $0))
BUILD_DIR=$TEST_DIR/build
BUILD_SCRIPT="cmake $SOURCE_DIR $CUDA_MODE -DPARALLEL_GRID_DIMENSION=3 -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DPRINT_MESSAGE=ON -DCXX11_ENABLED=ON; make fdtd3d"

$BASE_DIR/build-base.sh "$TEST_DIR" "$BUILD_DIR" "$BUILD_SCRIPT"
if [ $? -ne 0 ]; then
  exit 1
fi

exit 0

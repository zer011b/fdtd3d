#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

USED_MODE=$3

MODE="-DSOLVER_DIM_MODES=TMZ"
if [[ "$USED_MODE" -eq "1" ]]; then
  MODE="$MODE -DCUDA_ENABLED=ON -DCUDA_ARCH_SM_TYPE=sm_35"
elif [[ "$USED_MODE" -eq "2" ]]; then
  MODE="$MODE -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=x -DLINK_NUMA=ON -DPARALLEL_GRID_DIMENSION=2"
fi

if [[ "$USED_MODE" -eq "0" || "$USED_MODE" -eq "2" ]]; then
  MODE="$MODE -DCMAKE_BUILD_TYPE=ReleaseWithAsserts"
elif [[ "$USED_MODE" -eq "1" ]]; then
  MODE="$MODE -DCMAKE_BUILD_TYPE=Debug"
fi

TEST_DIR=$(dirname $(readlink -f $0))
BUILD_DIR=$TEST_DIR/build
BUILD_SCRIPT="cmake $SOURCE_DIR $MODE -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=OFF -DPRINT_MESSAGE=ON -DCXX11_ENABLED=ON; make fdtd3d"

$BASE_DIR/build-base.sh "$TEST_DIR" "$BUILD_DIR" "$BUILD_SCRIPT"
if [ $? -ne 0 ]; then
  exit 1
fi

exit 0

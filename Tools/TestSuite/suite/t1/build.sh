#!/bin/bash

BASE_DIR=$1
SOURCE_DIR=$2

TEST_DIR=$(dirname $(readlink -f $0))
BUILD_DIR=$TEST_DIR/build
BUILD_SCRIPT="cmake $SOURCE_DIR -DPARALLEL_GRID_DIMENSION=3 -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DTIME_STEPS=2 -DPRINT_MESSAGE=ON -DCXX11_ENABLED=ON; make fdtd3d"

$BASE_DIR/build-base.sh "$TEST_DIR" "$BUILD_DIR" "$BUILD_SCRIPT"
if [ $? -ne 0 ]; then
  exit 1
fi

g++ $TEST_DIR/exact.cpp -o $TEST_DIR/exact
if [ $? -ne 0 ]; then
  exit 1
fi

exit 0

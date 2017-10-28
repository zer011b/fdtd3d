#!/bin/bash

# This is the base build script for test suite. It launches build with specified command.

TEST_DIR="$1"
BUILD_DIR="$2"
BUILD_SCRIPT="$3"

CUR_DIR=`pwd`

mkdir -p $BUILD_DIR
cd $BUILD_DIR

eval $BUILD_SCRIPT &>/dev/null

if [ $? -ne 0 ]; then
  echo "Build failed"
  exit 1
fi

cp Source/fdtd3d $TEST_DIR/fdtd3d
cd $CUR_DIR

exit 0

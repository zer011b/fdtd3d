#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

rm fdtd3d
rm -rf build
rm *.bmp*

cd $CUR_DIR

exit $retval

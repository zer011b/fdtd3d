#!/bin/bash

set -ex

# Home directory of project where root CMakeLists.txt is placed
HOME_DIR=$1

# Build directory of unit test
BUILD_DIR=$2

# CXX compiler
CXX_COMPILER=$3

# C compiler
C_COMPILER=$4

# Build type
BUILD_TYPE=$5

# Whether build with cxx11 or not
CXX11_ENABLED=$6

# Whether use complex values or not
COMPLEX_FIELD_VALUES=$7

# Cuda device id
DEVICE=$8

# Cuda arch
ARCH=$9

# Whether to launch test or not
DO_LAUNCH=${10}

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  for VALUE_TYPE in f d; do

    cmake ${HOME_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DVALUE_TYPE=${VALUE_TYPE} \
      -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
      -DPRINT_MESSAGE=ON \
      -DCXX11_ENABLED=${CXX11_ENABLED} \
      -DCUDA_ENABLED=ON \
      -DCUDA_ARCH_SM_TYPE=${ARCH} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_C_COMPILER=${C_COMPILER}

    res=$(echo $?)

    if [[ res -ne 0 ]]; then
      exit 1
    fi

    make unit-test-internalscheme

    res=$(echo $?)

    if [[ res -ne 0 ]]; then
      exit 1
    fi

    if [[ DO_LAUNCH -ne 0 ]]; then
      ./Source/UnitTests/unit-test-internalscheme --time-steps 10 --point-source x:10,y:10,z:10 --point-source-ex \
        --cuda-gpus $DEVICE --num-cuda-threads x:4,y:4,z:4 --use-cuda

      if [[ "$?" -ne "0" ]]; then
        exit 1
      fi

      ./Source/UnitTests/unit-test-internalscheme --time-steps 10 --point-source x:10,y:10,z:10 --point-source-ex --use-ca-cb \
        --cuda-gpus $DEVICE --num-cuda-threads x:4,y:4,z:4 --use-cuda

      if [[ "$?" -ne "0" ]]; then
        exit 1
      fi

      ./Source/UnitTests/unit-test-internalscheme --time-steps 200 --use-tfsf \
        --cuda-gpus $DEVICE --num-cuda-threads x:4,y:4,z:4 --use-cuda

      if [[ "$?" -ne "0" ]]; then
        exit 1
      fi

      ./Source/UnitTests/unit-test-internalscheme --time-steps 200 --use-tfsf --use-ca-cb \
        --cuda-gpus $DEVICE --num-cuda-threads x:4,y:4,z:4 --use-cuda

      if [[ "$?" -ne "0" ]]; then
        exit 1
      fi
    fi

  done
}

build

exit 0

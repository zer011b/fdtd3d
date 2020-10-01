#!/bin/bash

set -ex

DO_LAUNCH=$1
shift

# Home directory of project where root CMakeLists.txt is placed
HOME_DIR=$1

# Build directory of unit test
BUILD_DIR=$2

# CXX compiler
CXX_COMPILER=$3

# C compiler
C_COMPILER=$4

CXX11_ENABLED=$5

# EX_HY EX_HZ EY_HX EY_HZ EZ_HX EZ_HX TEX TEY TEZ TMX TMY TMZ DIM3
MODE=$6

ARCH_SM_TYPE=$7

DEVICE=$8

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  for VALUE_TYPE in f d; do
    for COMPLEX_FIELD_VALUES in ON OFF; do
      for LARGE_COORDINATES in ON OFF; do

        cmake ${HOME_DIR} -DSOLVER_DIM_MODES=$MODE \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DVALUE_TYPE=${VALUE_TYPE} \
          -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
          -DPARALLEL_GRID_DIMENSION=3 \
          -DPRINT_MESSAGE=OFF \
          -DPARALLEL_GRID=OFF \
          -DPARALLEL_BUFFER_DIMENSION=x \
          -DCXX11_ENABLED=${CXX11_ENABLED} \
          -DCUDA_ENABLED=ON \
          -DCUDA_ARCH_SM_TYPE=$ARCH_SM_TYPE \
          -DLARGE_COORDINATES=${LARGE_COORDINATES} \
          -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
          -DCMAKE_C_COMPILER=${C_COMPILER} \
          -DDYNAMIC_GRID=OFF \
          -DCOMBINED_SENDRECV=OFF \
          -DMPI_CLOCK=OFF

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        make unit-test-internalscheme

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        if [[ DO_LAUNCH -eq 1 ]]; then
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
    done
  done
}

build

exit 0

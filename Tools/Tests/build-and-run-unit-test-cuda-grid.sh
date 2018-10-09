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

DEVICE=$6

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  for VALUE_TYPE in f d; do
    for TIME_STEPS in 1 2; do
      for COMPLEX_FIELD_VALUES in ON OFF; do
      for LARGE_COORDINATES in ON OFF; do

        if [ "${VALUE_TYPE}" == "ld" ] && [ "${COMPLEX_FIELD_VALUES}" == "ON" ]; then
          continue
        fi

        cmake ${HOME_DIR} -DCMAKE_BUILD_TYPE=ReleaseWithAsserts \
          -DVALUE_TYPE=${VALUE_TYPE} \
          -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
          -DTIME_STEPS=${TIME_STEPS} \
          -DPARALLEL_GRID_DIMENSION=3 \
          -DPRINT_MESSAGE=OFF \
          -DPARALLEL_GRID=OFF \
          -DPARALLEL_BUFFER_DIMENSION=x \
          -DCXX11_ENABLED=${CXX11_ENABLED} \
          -DCUDA_ENABLED=ON \
          -DCUDA_ARCH_SM_TYPE=sm_50 \
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

        make unit-test-cuda-grid

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        if [[ DO_LAUNCH -eq 1 ]]; then
          ./Tests/unit-test-cuda-grid $DEVICE

          res=$(echo $?)

          if [[ res -ne 0 ]]; then
            exit 1
          fi
        fi
      done
      done
    done
  done
}

build

exit 0

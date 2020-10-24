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

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  for VALUE_TYPE in f d ld; do
  for COMPLEX_FIELD_VALUES in ON OFF; do
  for MPI_CLOCK in ON OFF; do

    cmake ${HOME_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DVALUE_TYPE=${VALUE_TYPE} \
      -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
      -DPARALLEL_GRID_DIMENSION=3 \
      -DPRINT_MESSAGE=ON \
      -DPARALLEL_GRID=ON \
      -DPARALLEL_BUFFER_DIMENSION=x \
      -DCXX11_ENABLED=${CXX11_ENABLED} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DMPI_CLOCK=${MPI_CLOCK}

    res=$(echo $?)

    if [[ res -ne 0 ]]; then
      exit 1
    fi

    make unit-test-clock

    res=$(echo $?)

    if [[ res -ne 0 ]]; then
      exit 1
    fi

    ./Source/UnitTests/unit-test-clock

    res=$(echo $?)

    if [[ res -ne 0 ]]; then
      exit 1
    fi

  done
  done
  done
}

build

exit 0

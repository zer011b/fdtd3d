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

# Release, RelWithDebInfo, Debug
BUILD_MODE=$5

# ON or OFF
CXX11_ENABLED=$6

# ON or OFF
COMPLEX_FIELD_VALUES=$7

# ON or OFF
PRINT_MESSAGE=$8

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  PARALLEL_GRID_MODE=$1
  PARALLEL_GRID_DIM=$2
  LIST_OF_BUFFERS="$3"

  for VALUE_TYPE in f d ld; do
    for PARALLEL_BUFFER in `echo $LIST_OF_BUFFERS`; do
      if [ "${VALUE_TYPE}" == "ld" ] && [ "${COMPLEX_FIELD_VALUES}" == "ON" ]; then
        continue
      fi

      cmake ${HOME_DIR} -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
        -DVALUE_TYPE=${VALUE_TYPE} \
        -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
        -DPARALLEL_GRID_DIMENSION=${PARALLEL_GRID_DIM} \
        -DPRINT_MESSAGE=${PRINT_MESSAGE} \
        -DPARALLEL_GRID=${PARALLEL_GRID_MODE} \
        -DPARALLEL_BUFFER_DIMENSION=${PARALLEL_BUFFER} \
        -DCXX11_ENABLED=${CXX11_ENABLED} \
        -DCUDA_ENABLED=OFF \
        -DCUDA_ARCH_SM_TYPE=sm_50 \
        -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
        -DCMAKE_C_COMPILER=${C_COMPILER}

      # TODO: add next flags to testing:
      # -DDYNAMIC_GRID=
      # -DCOMBINED_SENDRECV
      # -DMPI_DYNAMIC_CLOCK

      res=$(echo $?)

      if [[ res -ne 0 ]]; then
        exit 1
      fi

      make fdtd3d

      res=$(echo $?)

      if [[ res -ne 0 ]]; then
        exit 1
      fi
    done
  done
}

array3D="x y z xy yz xz xyz"
build ON 3 "$array3D"
build OFF 3 "x"

exit 0

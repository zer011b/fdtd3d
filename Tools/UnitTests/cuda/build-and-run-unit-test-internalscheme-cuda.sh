#!/bin/bash

#
#  Copyright (C) 2020 Gleb Balykov
#
#  This file is part of fdtd3d.
#
#  fdtd3d is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  fdtd3d is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with fdtd3d; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

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

# Type of values
VALUE_TYPE=$8

# Cuda device id
DEVICE=$9

# Cuda arch
ARCH=${10}

# Whether to launch test or not
DO_LAUNCH=${11}

function build
{
  rm -rf ${BUILD_DIR}
  mkdir -p ${BUILD_DIR}
  pushd ${BUILD_DIR}

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

  popd
}

build

exit 0

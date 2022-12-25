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

# Toolchain file
TOOLCHAIN_FILE_PATH=$9
TOOLCHAIN=""
if [[ "$TOOLCHAIN_FILE_PATH" != "" ]]; then
  echo "Testing cross build"
  TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE_PATH"
fi

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
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    ${TOOLCHAIN}

  res=$(echo $?)

  if [[ res -ne 0 ]]; then
    exit 1
  fi

  make unit-test-dumpers-loaders

  res=$(echo $?)

  if [[ res -ne 0 ]]; then
    exit 1
  fi

  if [[ "$TOOLCHAIN_FILE_PATH" != "" ]]; then
    file ./Source/UnitTests/unit-test-dumpers-loaders
    sudo rm -rf ${ROOTFS}/fdtd3d
    sudo mkdir -p ${ROOTFS}/fdtd3d
    sudo cp ./Source/UnitTests/unit-test-dumpers-loaders ${ROOTFS}/fdtd3d
    sudo chroot ${ROOTFS} /fdtd3d/unit-test-dumpers-loaders
  else
    ./Source/UnitTests/unit-test-dumpers-loaders
  fi

  res=$(echo $?)

  if [[ res -ne 0 ]]; then
    exit 1
  fi

  popd
}

build

exit 0

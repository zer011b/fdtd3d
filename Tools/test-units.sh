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

# Test all unit tests for native arch (i.e. no cross-arch/cuda launch)
#
# To test all unit tests:
#   ./test-units.sh <home_dir> <build_dir> "" "" "" "" "" "" ""
#
# To test scpecific unit test in all configurations
#   ./test-units.sh <home_dir> <build_dir> "<test_sh_path>" "" "" "" "" "" ""

set -ex

# Home directory of project where root CMakeLists.txt is located
HOME_DIR=$1; shift
if [[ "$HOME_DIR" == "" ]]; then
  echo "HOME_DIR is required"
  exit 1
fi

# Build directory of unit test
BUILD_DIR=$1; shift
if [[ "$BUILD_DIR" == "" ]]; then
  echo "BUILD_DIR is required"
  exit 1
fi

SCRIPTS_VALUES="$1"; shift
if [[ "$SCRIPTS_VALUES" == "" ]]; then
  SCRIPTS_VALUES=$(find ${HOME_DIR}/Tools/UnitTests/native/ -name "*.sh")
fi

COMPILERS_VALUES="$1"; shift
if [[ "$COMPILERS_VALUES" == "" ]]; then
  COMPILERS_VALUES="gcc,g++"
fi

CMAKE_BUILD_TYPE_VALUES="$1"; shift
if [[ "$CMAKE_BUILD_TYPE_VALUES" == "" ]]; then
  CMAKE_BUILD_TYPE_VALUES="RelWithDebInfo Debug"
fi

CXX11_ENABLED_VALUES="$1"; shift
if [[ "$CXX11_ENABLED_VALUES" == "" ]]; then
  CXX11_ENABLED_VALUES="ON OFF"
fi

COMPLEX_FIELD_VALUES_VALUES="$1"; shift
if [[ "$COMPLEX_FIELD_VALUES_VALUES" == "" ]]; then
  COMPLEX_FIELD_VALUES_VALUES="ON OFF"
fi

VALUE_TYPE_VALUES="$1"; shift
if [[ "$VALUE_TYPE_VALUES" == "" ]]; then
  VALUE_TYPE_VALUES="f d ld"
fi

TOOLCHAIN_FILE_PATH="$1"; shift

for SCRIPT in $SCRIPTS_VALUES; do
for COMPILERS in $COMPILERS_VALUES; do
for CMAKE_BUILD_TYPE in $CMAKE_BUILD_TYPE_VALUES; do
for CXX11_ENABLED in $CXX11_ENABLED_VALUES; do
for COMPLEX_FIELD_VALUES in $COMPLEX_FIELD_VALUES_VALUES; do
for VALUE_TYPE in $VALUE_TYPE_VALUES; do

  CMAKE_C_COMPILER=$(echo $COMPILERS | awk -F ',' '{print $1}')
  CMAKE_CXX_COMPILER=$(echo $COMPILERS | awk -F ',' '{print $2}')

  CUDA_DEVICE_ID=$(echo $CUDA_DEVICE | awk -F ',' '{print $1}')
  CUDA_DEVICE_ARCH=$(echo $CUDA_DEVICE | awk -F ',' '{print $2}')

  ${SCRIPT} ${HOME_DIR} ${BUILD_DIR} ${CMAKE_CXX_COMPILER} ${CMAKE_C_COMPILER} ${CMAKE_BUILD_TYPE} ${CXX11_ENABLED} ${COMPLEX_FIELD_VALUES} ${VALUE_TYPE} ${TOOLCHAIN_FILE_PATH}

  res=$(echo $?)
  if [[ res -ne 0 ]]; then
    echo "Unit test failed"
    exit 2
  fi

  echo "Unit test successful"

done
done
done
done
done
done

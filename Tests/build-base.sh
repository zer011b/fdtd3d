#!/bin/bash

# This is the base build script for test suite. It launches build with specified command.

#
#  Copyright (C) 2017 Gleb Balykov
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

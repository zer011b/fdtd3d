#!/bin/bash

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

TEST=$1; shift

# 1 to test MPI
TEST_MPI=$1; shift
# 1 to test in CUDA mode
CUDA_MODE=$1; shift

ARGS=$@

CUR_DIR=`pwd`

# check exit code
function check_res ()
{
  if [ $? -ne 0 ]; then
    exit 1
  fi
}

if [ "$CUDA_MODE" == "1" ]; then
  echo "TESTING GPU"
  $CUR_DIR/Tests/suite/$TEST/build.sh 1 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh 1 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING GPU+PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh 3 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh 3 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
else
  echo "TESTING SEQUENTIAL"
  $CUR_DIR/Tests/suite/$TEST/build.sh 0 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh 0 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh 2 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh 2 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
fi

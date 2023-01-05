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

set -e

export LC_NUMERIC=C

USED_MODE=$1
BASE_DIR=$2
SOURCE_DIR=$3
C_COMPILER=$4
CXX_COMPILER=$5
TOOLCHAIN=$6

MODE=""
RUNNER=""

if [[ "$USED_MODE" -eq "1" ]]; then
  MODE="$MODE --use-cuda --cuda-gpus 0 --num-cuda-threads x:4,y:4,z:4"
fi
if [[ "$USED_MODE" -eq "3" ]]; then
  MODE="$MODE --use-cuda --cuda-gpus 0,0 --cuda-buffer-size 2 --buffer-size 2 --num-cuda-threads x:4,y:4,z:4"
fi
if [[ "$USED_MODE" -eq "2" || "$USED_MODE" -eq "3" ]]; then
  MODE="$MODE"
  RUNNER="mpirun -n 2 --oversubscribe"
fi

if [[ "$TOOLCHAIN" != "" ]]; then
  RUNNER="sudo chroot ${ROOTFS} $RUNNER"
  RUNPATH="/fdtd3d/"
else
  RUNNER="$RUNNER"
  RUNPATH="./"
fi

accuracy_percent="0.0001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"
  local layout_type="$4"

  local lambda="0.02"

  local ret=$((0))

  output_file=$(mktemp /tmp/fdtd3d.$size.$dx.XXXXXXXX)

  $RUNNER "$RUNPATH"fdtd3d $MODE --time-steps $timesteps --size x:$size --same-size --3d --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --use-polinom3-border-condition --use-polinom3-start-values \
    --use-polinom3-right-side --calc-polinom3-diff-norm --layout-type $layout_type &> $output_file

  local max_diff=$(cat $output_file | grep "DIFF NORM Ez" | awk '{if (max < $14) {max = $14}} END{printf "%.20f", max}')
  local is_ok=$(echo $max_diff $accuracy_percent | awk '{if ($1 > $2) {print 0} else {print 1}}')
  if [ "$is_ok" != "1" ]; then
    echo "Failed $size $dx"
    ret=$((1))
  fi

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

if [[ "$TOOLCHAIN" != "" ]]; then
  sudo rm -rf ${ROOTFS}/fdtd3d
  sudo mkdir -p ${ROOTFS}/fdtd3d
  sudo cp ./fdtd3d ${ROOTFS}/fdtd3d/
fi

retval=$((0))

size1="20"
dx1="0.001"

size2="40"
dx2="0.0005"

launch $size1 101 $dx1 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size1 101 $dx1 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size2 201 $dx2 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size2 201 $dx2 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

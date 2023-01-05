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
  RUNNER_NATIVE="sudo chroot ${ROOTFS}"
  RUNPATH="/fdtd3d/"
  RESPATH="${ROOTFS}/"
  RUNNER_RES="sudo"
else
  RUNNER="$RUNNER"
  RUNNER_NATIVE=""
  RUNPATH="./"
  RESPATH="./"
  RUNNER_RES=""
fi

timestep="30"
accuracy="0.0000001"

function launch ()
{
  local binary_prefix=$1
  local layout_type="$2"

  output_file=$(mktemp /tmp/fdtd3d.vacuum3D.XXXXXXXX)

  tmp_test_file=$($RUNNER_NATIVE mktemp /tmp/vacuum2D_planewave_TMz.XXXXXXXX.txt)
  cp vacuum2D_planewave_TMz.txt tmp.txt
  echo $MODE >> tmp.txt
  echo "--layout-type $layout_type" >> tmp.txt
  $RUNNER_RES cp tmp.txt "$ROOTFS"$tmp_test_file

  $RUNNER "$RUNPATH"fdtd3d${binary_prefix} --cmd-from-file $tmp_test_file &> $output_file

  $RUNNER_RES mv "$RESPATH"previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt "$RESPATH"previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt.$binary_prefix

  local ret=$?

  return $ret
}

function cmp ()
{
  local filename_complex="$1"
  local filename_real="$2"

  while read -r line
  do
      x=$(echo $line | awk '{print $1}')
      y=$(echo $line | awk '{print $2}')
      val_real=$(echo $line | awk '{print $3}')
      val_imag=$(echo $line | awk '{print $4}')

      val_real_build=$(cat $filename_real | grep "^$x $y " | awk '{print $3}')

      is_ok=$(echo $val_real $val_real_build $accuracy | awk '{if (($1-$2)*($1-$2) < $3) {print "OK"} else {print "FAIL"} }')

      if [[ "$is_ok" != "OK" ]]; then
        echo "Mismatch between complex and real modes: $x $y, $val_real, $val_real_build"
        retval=$((2))
        break
      fi
  done < "$filename_complex"
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

if [[ "$TOOLCHAIN" != "" ]]; then
  sudo rm -rf ${ROOTFS}/fdtd3d
  sudo mkdir -p ${ROOTFS}/fdtd3d
  sudo cp ./fdtd3d_complex ${ROOTFS}/fdtd3d/
  sudo cp ./fdtd3d_real ${ROOTFS}/fdtd3d/
fi

filename_complex="previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt._complex"
filename_real="previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt._real"

retval=$((0))

launch _complex 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch _real 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi

$RUNNER_RES cp "$RESPATH"$filename_complex ./txt._complex
$RUNNER_RES cp "$RESPATH"$filename_real ./txt._real
cmp ./txt._complex ./txt._real
$RUNNER_RES rm "$RESPATH"previous-*

launch _complex 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch _real 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

$RUNNER_RES cp "$RESPATH"$filename_complex ./txt._complex
$RUNNER_RES cp "$RESPATH"$filename_real ./txt._real
cmp ./txt._complex ./txt._real
$RUNNER_RES rm "$RESPATH"previous-*

cd $CUR_DIR

exit $retval

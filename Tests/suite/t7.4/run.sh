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

function launch ()
{
  local num_time_steps=$1
  local sizez=$2
  local dz=$3
  local wavelen=$4
  local pmlsize=$5
  local tfsfsize=$6
  local sphere_center=$7
  local sphere_radius=$8

  local norm_start=$9
  local norm_end=${10}
  local exp_type=${11}

  local accuracy_percent=${12}

  local eps_sphere=${13}
  local layout_type="${14}"

  time_step_before=$((num_time_steps - 1))

  output_file=$(mktemp /tmp/fdtd3d.vacuum3D.XXXXXXXX)

  $RUNNER "$RUNPATH"fdtd3d $MODE --time-steps $num_time_steps --size x:$sizez --1d-eyhz --angle-teta 90 --angle-phi 0 --angle-psi 0 \
    --dx $dz --wavelength $4 --courant-factor 0.5 --log-level 2 --pml-size x:$pmlsize --use-pml \
    --use-tfsf --tfsf-size-left x:$tfsfsize --tfsf-size-right x:0 \
    --eps-sphere $eps_sphere --eps-sphere-center x:$sphere_center --eps-sphere-radius $sphere_radius \
    --norm-start x:$norm_start --norm-end x:$norm_end --calc-${exp_type}-eyhz-diff-norm --layout-type $layout_type &> $output_file

  local ret=$?

  local max_diff_real=$(cat $output_file | grep "Timestep $time_step_before" | awk '{if (max < $16) {max = $16}} END{printf "%.20f", max}')
  local max_diff_imag=$(cat $output_file | grep "Timestep $time_step_before" | awk '{if (max < $19) {max = $19}} END{printf "%.20f", max}')
  local max_diff_mod=$(cat $output_file | grep "Timestep $time_step_before" | awk '{if (max < $27) {max = $27}} END{printf "%.20f", max}')

  #echo "!!! $max_diff_real, $max_diff_imag, $max_diff_mod"

  local is_ok=$(echo $max_diff_real $accuracy_percent | awk '{if ($1 > $2) {print 0} else {print 1}}')
  if [ "$is_ok" != "1" ]; then
    echo "Failed real"
    ret=$((1))
  fi

  local is_ok=$(echo $max_diff_imag $accuracy_percent | awk '{if ($1 > $2) {print 0} else {print 1}}')
  if [ "$is_ok" != "1" ]; then
    echo "Failed imag"
    ret=$((1))
  fi

  local is_ok=$(echo $max_diff_mod $accuracy_percent | awk '{if ($1 > $2) {print 0} else {print 1}}')
  if [ "$is_ok" != "1" ]; then
    echo "Failed mod"
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
launch 10000 600 0.0001 0.02 50 150 450 150 155 545 exp1 0.12 1 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch 10000 600 0.0001 0.02 50 150 450 150 155 545 exp1 0.12 1 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch 10000 600 0.0001 0.02 50 150 450 150 305 545 exp3 0.12 4 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch 10000 600 0.0001 0.02 50 150 450 150 305 545 exp3 0.12 4 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch 10000 600 0.0001 0.02 50 150 450 150 55 145 exp2 0.17 4 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch 10000 600 0.0001 0.02 50 150 450 150 55 145 exp2 0.17 4 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

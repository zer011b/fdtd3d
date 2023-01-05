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

accuracy_exact="0.000001"
accuracy_exact_mod="0.0000000000001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"
  local layout_type="$4"

  local lambda="0.02"
  local length=$(echo $timesteps | awk '{print $1 - 10}')

  $RUNNER "$RUNPATH"./fdtd3d $MODE --time-steps $timesteps --size x:$size --same-size --2d-tez --angle-phi 90 --angle-psi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --save-res --save-tfsf-e-incident --save-as-txt --use-tfsf --tfsf-size-left x:4 --tfsf-size-right x:4 \
    --same-size-tfsf --courant-factor 1.0 --layout-type $layout_type &>/dev/null

  local ret=$((0))

  for line_num in `seq 1 1 $length`; do
    # exact value
    local i=$(echo $line_num | awk '{print $1 - 1}')
    local n=$(echo $timesteps | awk '{print $1 - 1}')

    local exact=$($RUNNER_NATIVE "$RUNPATH"exact $lambda $dx $i $n 1.0)

    local exact_val_re=$(echo $exact | awk '{printf "%.17g", $1}')
    local exact_val_im=$(echo $exact | awk '{printf "%.17g", $2}')
    local exact_val_mod=$(echo $exact | awk '{printf "%.17g", $3}')

    # numerical value
    local line=$($RUNNER_RES sed "${line_num}q;d" "$RESPATH"previous-1_\[timestep\=$timesteps\]_\[pid\=0\]_\[name\=EInc\].txt)
    local index=$(echo $line | awk '{printf "%.17g", $1}')
    local numerical_val_re=$(echo $line | awk '{printf "%.17g", $2}')
    local numerical_val_im=$(echo $line | awk '{printf "%.17g", $3}')
    local numerical_val_mod=$(echo $numerical_val_re $numerical_val_im | awk '{printf "%.17g", sqrt($1 * $1 + $2 * $2)}')

    if [ $i -ne $index ]; then
      echo "Incorrect output from fdtd3d"
      ret=$((1))
      break
    fi

    is_correct=$(echo $exact_val_re $exact_val_im $exact_val_mod \
                      $numerical_val_re $numerical_val_im $numerical_val_mod \
                      $accuracy_exact $accuracy_exact_mod \
                 | awk 'function abs(v) {return v < 0 ? -v : v}
                        {
                          if (abs($1 - $4) > $7 || abs($2 - $5) > $7 || abs($3 - $6) > $8 || abs($3 - 1.0) > $8)
                          {
                            print 0;
                          }
                          else
                          {
                            print 1;
                          }
                        }')
    if [ "$is_correct" != "1" ]; then
      echo "Line $line_num failed: exact($exact_val_re, $exact_val_im, $exact_val_mod), " \
           "numerical($numerical_val_re, $numerical_val_im, $numerical_val_mod)"
      ret=$((2))
      break
    fi
  done

  $RUNNER_RES rm "$RESPATH"previous-*

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

if [[ "$TOOLCHAIN" != "" ]]; then
  sudo rm -rf ${ROOTFS}/fdtd3d
  sudo mkdir -p ${ROOTFS}/fdtd3d
  sudo cp ./fdtd3d ${ROOTFS}/fdtd3d/
  sudo cp ./exact ${ROOTFS}/fdtd3d/
fi

size="12"
retval=$((0))

launch $size 251 0.0004 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size 251 0.0004 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size 501 0.0002 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size 501 0.0002 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

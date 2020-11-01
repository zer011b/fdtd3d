#!/bin/bash

set -e

USED_MODE=$1
BASE_DIR=$2
SOURCE_DIR=$3

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
  RUNNER="mpirun -n 2"
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

  $RUNNER ./fdtd3d $MODE --time-steps $num_time_steps --size y:$sizez --1d-ezhx --angle-teta 90 --angle-phi 90 --angle-psi 90 \
    --dx $dz --wavelength $4 --courant-factor 0.5 --log-level 2 --pml-size y:$pmlsize --use-pml \
    --use-tfsf --tfsf-size-left y:$tfsfsize --tfsf-size-right y:0 \
    --eps-sphere $eps_sphere --eps-sphere-center y:$sphere_center --eps-sphere-radius $sphere_radius \
    --norm-start y:$norm_start --norm-end y:$norm_end --calc-${exp_type}-ezhx-diff-norm --layout-type $layout_type &> $output_file

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

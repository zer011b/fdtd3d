#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

USED_MODE=$3

MODE=""
RUNNER=""
if [[ "$USED_MODE" -eq "1" ]]; then
  MODE="$MODE --use-cuda --cuda-gpus 0 --num-cuda-threads-x 4 --num-cuda-threads-y 4 --num-cuda-threads-z 4"
fi
if [[ "$USED_MODE" -eq "3" ]]; then
  MODE="$MODE --use-cuda --cuda-gpus 0,0 --cuda-buffer-size 2 --buffer-size 2 --num-cuda-threads-x 4 --num-cuda-threads-y 4 --num-cuda-threads-z 4"
fi
if [[ "$USED_MODE" -eq "2" || "$USED_MODE" -eq "3" ]]; then
  MODE="$MODE"
  RUNNER="mpirun -n 2"
fi

accuracy_percent="0.00001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"
  local layout_type="$4"

  local lambda="0.02"

  output_file=$(mktemp /tmp/fdtd3d.$size.$dx.XXXXXXXX)

  $RUNNER ./fdtd3d $MODE --time-steps $timesteps --sizex $size --same-size --3d --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --use-sin1-border-condition --use-sin1-start-values --calc-sin1-diff-norm \
    --eps-normed --mu-normed --courant-factor 149896229 --layout-type $layout_type &> $output_file

  local ret=$?

  max_diff=$(cat $output_file | grep "DIFF NORM Ez" | awk '{if ($10>max){max=$10}}END{print max}')

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

retval=$((0))

size1="10"
dx1="0.001"

size2="20"
dx2="0.0005"

size3="40"
dx3="0.00025"

function test ()
{
  local timesteps=$1
  local layout_type="$2"
  local ret=$((0))

  timesteps1=$(echo $timesteps | awk '{print $1 + 1}')
  launch $size1 $timesteps1 $dx1 $layout_type
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi
  diff1=$(echo $max_diff)

  timesteps2=$(echo $timesteps | awk '{print $1 * 2 + 1}')
  launch $size2 $timesteps2 $dx2 $layout_type
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi
  diff2=$(echo $max_diff)

  timesteps3=$(echo $timesteps | awk '{print $1 * 4 + 1}')
  launch $size3 $timesteps3 $dx3 $layout_type
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi
  diff3=$(echo $max_diff)

  is_ok=$(echo $diff1 $diff2 $diff3 | awk '
            {
              d1 = $1 / $2;
              d2 = $2 / $3;
              if (3.3 < d1 && 3.3 < d2)
              {
                print 1;
              }
              else
              {
                print 0;
              }
            }')
  if [ "$is_ok" != "1" ]; then
    echo "Second order of accuracy failed: $diff1 $diff2 $diff3"
    ret=$((2))
  fi

  return $ret
}

test 40 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
test 40 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

test 80 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
test 80 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

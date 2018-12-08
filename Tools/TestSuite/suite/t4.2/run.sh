#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

USE_CUDA=$3

CUDA_MODE=""
if [[ "$USE_CUDA" -eq "1" ]]; then
  CUDA_MODE="--num-cuda-gpus 0 --num-cuda-threads-x 4 --num-cuda-threads-y 4 --num-cuda-threads-z 4"
fi

accuracy_percent="0.0001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"

  local lambda="0.02"

  local ret=$((0))

  output_file=$(mktemp /tmp/fdtd3d.$size.$dx.XXXXXXXX)

  ./fdtd3d $CUDA_MODE --time-steps $timesteps --sizex $size --same-size --2d-tmz --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --use-polinom3-border-condition --use-polinom3-start-values \
    --use-polinom3-right-side --calc-polinom3-diff-norm &> $output_file

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

retval=$((0))

size1="20"
dx1="0.001"

size2="40"
dx2="0.0005"

launch $size1 101 $dx1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size2 201 $dx2
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

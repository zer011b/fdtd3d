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

function launch ()
{
  output_file=$(mktemp /tmp/fdtd3d.vacuum1D.XXXXXXXX)

  tmp_test_file=$(mktemp /tmp/vacuum1D_ExHz.XXXXXXXX.txt)
  cp ${SOURCE_DIR}/Examples/vacuum1D_ExHz.txt $tmp_test_file
  echo $MODE >> $tmp_test_file

  $RUNNER ./fdtd3d --cmd-from-file $tmp_test_file &> $output_file

  local ret=$?

  val_max=$(cat previous-1_[timestep=100]_[pid=0]_[name=Ex]_[mod].txt | awk '{print $1}')
  val_min=$(cat previous-1_[timestep=100]_[pid=0]_[name=Ex]_[mod].txt | awk '{print $2}')
  is_ok=$(echo $val_max $val_min | awk '
            {
              if (1.01 <= $1 && $1 <= 1.02 && 0.0 == $2)
              {
                print 1;
              }
              else
              {
                print 0;
              }
            }')
  if [ "$is_ok" != "1" ]; then
    echo "Test result is incorrect for Examples/vacuum1D_ExHz.txt"
    ret=$((2))
  fi

  if [ "$ret" != "0" ]; then
    return $ret
  fi

  cp ${SOURCE_DIR}/Examples/vacuum1D_ExHz_scattered.txt $tmp_test_file
  echo $MODE >> $tmp_test_file

  $RUNNER ./fdtd3d --cmd-from-file $tmp_test_file &> $output_file

  ret=$?

  val_max=$(cat previous-1_[timestep=100]_[pid=0]_[name=Ex]_[mod].txt | awk '{print $1}')
  val_min=$(cat previous-1_[timestep=100]_[pid=0]_[name=Ex]_[mod].txt | awk '{print $2}')
  is_ok=$(echo $val_max $val_min | awk '
            {
              if (0.0 <= $1 && $1 <= 1e-14 && 0.0 == $2)
              {
                print 1;
              }
              else
              {
                print 0;
              }
            }')
  if [ "$is_ok" != "1" ]; then
    echo "Test result is incorrect for Examples/vacuum1D_ExHz_scattered.txt"
    ret=$((2))
  fi

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

retval=$((0))
launch
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

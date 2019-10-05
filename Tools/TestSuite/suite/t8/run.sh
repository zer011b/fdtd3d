#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

USED_MODE=$3

MODE=""
RUNNER=""
if [[ "$USED_MODE" -eq "1" ]]; then
  MODE="--use-cuda --cuda-gpus 0 --num-cuda-threads-x 4 --num-cuda-threads-y 4 --num-cuda-threads-z 4"
elif [[ "$USED_MODE" -eq "2" ]]; then
  MODE=""
  RUNNER="mpirun -n 2"
fi

timestep="30"
accuracy="0.0000001"

function launch ()
{
  local binary_prefix=$1

  output_file=$(mktemp /tmp/fdtd3d.vacuum3D.XXXXXXXX)

  tmp_test_file=$(mktemp /tmp/vacuum2D_planewave_TMz.XXXXXXXX.txt)
  cp vacuum2D_planewave_TMz.txt $tmp_test_file
  echo $MODE >> $tmp_test_file

  $RUNNER ./fdtd3d${binary_prefix} --cmd-from-file $tmp_test_file &> $output_file

  mv previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt.$binary_prefix

  local ret=$?

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

retval=$((0))
launch _complex
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch _real
if [ $? -ne 0 ]; then
  retval=$((1))
fi

filename_complex="previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt._complex"
filename_real="previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt._real"

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

cd $CUR_DIR

exit $retval

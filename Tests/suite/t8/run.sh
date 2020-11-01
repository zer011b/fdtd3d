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

timestep="30"
accuracy="0.0000001"

function launch ()
{
  local binary_prefix=$1
  local layout_type="$2"

  output_file=$(mktemp /tmp/fdtd3d.vacuum3D.XXXXXXXX)

  tmp_test_file=$(mktemp /tmp/vacuum2D_planewave_TMz.XXXXXXXX.txt)
  cp vacuum2D_planewave_TMz.txt $tmp_test_file
  echo $MODE >> $tmp_test_file
  echo "--layout-type $layout_type" >> $tmp_test_file

  $RUNNER ./fdtd3d${binary_prefix} --cmd-from-file $tmp_test_file &> $output_file

  mv previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt previous-1_[timestep=$timestep]_[pid=0]_[name=Ez].txt.$binary_prefix

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

cmp $filename_complex $filename_real

launch _complex 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch _real 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cmp $filename_complex $filename_real

cd $CUR_DIR

exit $retval

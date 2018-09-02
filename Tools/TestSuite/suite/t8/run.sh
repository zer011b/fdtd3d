#!/bin/bash

set -e

BASE_DIR=$1
SOURCE_DIR=$2

timestep="30"
accuracy="0.0000001"

function launch ()
{
  local binary_prefix=$1

  output_file=$(mktemp /tmp/fdtd3d.vacuum3D.XXXXXXXX)

  ./fdtd3d${binary_prefix} --cmd-from-file vacuum2D_planewave_TMz.txt &> $output_file

  mv current[$timestep]_rank-0_Ez.txt current[$timestep]_rank-0_Ez.txt.$binary_prefix

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

filename_complex="current[$timestep]_rank-0_Ez.txt._complex"
filename_real="current[$timestep]_rank-0_Ez.txt._real"

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

#!/bin/bash

BASE_DIR=$1
SOURCE_DIR=$2

accuracy_percent="0.00001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"

  local lambda="0.02"

  local ret=$((0))

  ./fdtd3d --time-steps $timesteps --sizex $size --same-size --3d --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --use-sin1-border-condition --use-sin1-start-values --calc-sin1-diff-norm &> /tmp/$size.$dx.txt

  max_diff=$(cat /tmp/$size.$dx.txt | grep "DIFF NORM Ez" | awk '{if ($10>max){max=$10}}END{print max}')

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

size3="80"
dx3="0.00025"

function test ()
{
  timesteps=$1
  local ret=$((0))

  timesteps1=$(echo $timesteps | awk '{print $1 + 1}')
  launch $size1 $timesteps1 $dx1
  diff1=$(echo $max_diff)
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi

  timesteps2=$(echo $timesteps | awk '{print $1 * 2 + 1}')
  launch $size2 $timesteps2 $dx2
  diff2=$(echo $max_diff)
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi

  timesteps3=$(echo $timesteps | awk '{print $1 * 4 + 1}')
  launch $size3 $timesteps3 $dx3
  diff3=$(echo $max_diff)
  if [ $? -ne 0 ]; then
    ret=$((1))
  fi

  is_ok=$(echo $diff1 $diff2 $diff3 | awk '
            {
              d1 = $1 / $2;
              d2 = $2 / $3;
              if (4.0 < d1 && 4.0 < d2)
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

test 100
if [ $? -ne 0 ]; then
  retval=$((1))
fi

test 200
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

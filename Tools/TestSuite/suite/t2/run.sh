#!/bin/bash

BASE_DIR=$1
SOURCE_DIR=$2

function launch ()
{
  size="$1"
  timesteps="$2"
  dx="$3"

  lambda="0.02"

  ./fdtd3d --time-steps $timesteps --sizex $size --same-size --3d --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --use-test-border-condition --use-test-start-values --calc-test-diff-norm &> /tmp/$size.$dx.txt
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

retval=$((0))

size1="20"
dx1="0.0004"

size2="40"
dx2="0.0002"

launch $size1 51 $dx1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size2 101 $dx2
if [ $? -ne 0 ]; then
  retval=$((1))
fi

# compare norm

length=$(echo $size1 | awk '{print $1 - 1}')
for timestep in `seq 1 1 $length`; do
  timestep1=$(echo $timestep | awk '{print $1}')
  normRe_1=$(cat /tmp/$size1.$dx1.txt | grep "TEST: $timestep1 " | awk '{print $3}')
  normIm_1=$(cat /tmp/$size1.$dx1.txt | grep "TEST: $timestep1 " | awk '{print $4}')
  normMod_1=$(cat /tmp/$size1.$dx1.txt | grep "TEST: $timestep1 " | awk '{print $5}')

  timestep2=$(echo $timestep | awk '{print $1 * 2}')
  normRe_2=$(cat /tmp/$size2.$dx2.txt | grep "TEST: $timestep2 " | awk '{print $3}')
  normIm_2=$(cat /tmp/$size2.$dx2.txt | grep "TEST: $timestep2 " | awk '{print $4}')
  normMod_2=$(cat /tmp/$size2.$dx2.txt | grep "TEST: $timestep2 " | awk '{print $5}')

  echo "$normRe_1 vs $normRe_2 | $normMod_1 vs $normMod_2"
done

#rm fdtd3d exact

cd $CUR_DIR

exit $retval

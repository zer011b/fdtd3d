#!/bin/bash

BASE_DIR=$1
SOURCE_DIR=$2

accuracy_exact="0.000001"
accuracy_exact_mod="0.0000000000001"

function launch ()
{
  size="$1"
  timesteps="$2"
  dx="$3"

  lambda="0.02"
  length=$(echo $timesteps | awk '{print $1 - 10}')

  ./fdtd3d --time-steps $timesteps --sizex $size --same-size --3d --angle-phi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --save-res --save-tfsf-e-incident --save-as-txt --use-tfsf --tfsf-sizex 2 --same-size-tfsf \
    --courant-factor 1.0 &>/dev/null

  retval=$((0))

  for line_num in `seq 1 1 $length`; do
    # exact value
    i=$(echo $line_num | awk '{print $1 - 1}')
    n=$(echo $timesteps | awk '{print $1 - 1}')

    exact=$(./exact $lambda $dx $i $n 1.0)

    exact_val_re=$(echo $exact | awk '{printf "%.17g", $1}')
    exact_val_im=$(echo $exact | awk '{printf "%.17g", $2}')
    exact_val_mod=$(echo $exact | awk '{printf "%.17g", $3}')

    # numerical value
    line=$(sed "${line_num}q;d" current\[$timesteps\]_rank-0_EInc.txt)
    index=$(echo $line | awk '{printf "%.17g", $1}')
    numerical_val_re=$(echo $line | awk '{printf "%.17g", $2}')
    numerical_val_im=$(echo $line | awk '{printf "%.17g", $3}')
    numerical_val_mod=$(echo $numerical_val_re $numerical_val_im | awk '{printf "%.17g", sqrt($1 * $1 + $2 * $2)}')

    if [ $i -ne $index ]; then
      echo "Incorrect output from fdtd3d"
      retval=$((1))
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
      retval=$((2))
      break
    fi
  done

  rm current\[*

  return $retval
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

size="10"
retval=$((0))

launch $size 501 0.0004
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size 1001 0.0002
if [ $? -ne 0 ]; then
  retval=$((1))
fi

rm fdtd3d exact

cd $CUR_DIR

exit $retval

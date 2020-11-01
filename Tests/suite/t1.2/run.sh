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

accuracy_exact="0.000001"
accuracy_exact_mod="0.0000000000001"

function launch ()
{
  local size="$1"
  local timesteps="$2"
  local dx="$3"
  local layout_type="$4"

  local lambda="0.02"
  local length=$(echo $timesteps | awk '{print $1 - 10}')

  $RUNNER ./fdtd3d $MODE --time-steps $timesteps --size x:$size --same-size --2d-tez --angle-phi 90 --angle-psi 0 --dx $dx --wavelength $lambda \
    --log-level 0 --save-res --save-tfsf-e-incident --save-as-txt --use-tfsf --tfsf-size-left x:4 --tfsf-size-right x:4 \
    --same-size-tfsf --courant-factor 1.0 --layout-type $layout_type &>/dev/null

  local ret=$((0))

  for line_num in `seq 1 1 $length`; do
    # exact value
    local i=$(echo $line_num | awk '{print $1 - 1}')
    local n=$(echo $timesteps | awk '{print $1 - 1}')

    local exact=$(./exact $lambda $dx $i $n 1.0)

    local exact_val_re=$(echo $exact | awk '{printf "%.17g", $1}')
    local exact_val_im=$(echo $exact | awk '{printf "%.17g", $2}')
    local exact_val_mod=$(echo $exact | awk '{printf "%.17g", $3}')

    # numerical value
    local line=$(sed "${line_num}q;d" previous-1_\[timestep\=$timesteps\]_\[pid\=0\]_\[name\=EInc\].txt)
    local index=$(echo $line | awk '{printf "%.17g", $1}')
    local numerical_val_re=$(echo $line | awk '{printf "%.17g", $2}')
    local numerical_val_im=$(echo $line | awk '{printf "%.17g", $3}')
    local numerical_val_mod=$(echo $numerical_val_re $numerical_val_im | awk '{printf "%.17g", sqrt($1 * $1 + $2 * $2)}')

    if [ $i -ne $index ]; then
      echo "Incorrect output from fdtd3d"
      ret=$((1))
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
      ret=$((2))
      break
    fi
  done

  rm previous-*

  return $ret
}

CUR_DIR=`pwd`
TEST_DIR=$(dirname $(readlink -f $0))
cd $TEST_DIR

size="12"
retval=$((0))

launch $size 251 0.0004 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size 251 0.0004 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

launch $size 501 0.0002 0
if [ $? -ne 0 ]; then
  retval=$((1))
fi
launch $size 501 0.0002 1
if [ $? -ne 0 ]; then
  retval=$((1))
fi

cd $CUR_DIR

exit $retval

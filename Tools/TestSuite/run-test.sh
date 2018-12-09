#!/bin/bash

set -ex

TEST=$1
shift

ARGS=$@

CUR_DIR=`pwd`

# check exit code
function check_res ()
{
  if [ $? -ne 0 ]; then
    exit 1
  fi
}

echo "TESTING SEQUENTIAL"
$CUR_DIR/Tools/TestSuite/suite/$TEST/build.sh $ARGS 0
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/run.sh $ARGS 0
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/cleanup.sh $ARGS
check_res

echo "TESTING PARALLEL"
$CUR_DIR/Tools/TestSuite/suite/$TEST/build.sh $ARGS 2
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/run.sh $ARGS 2
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/cleanup.sh $ARGS
check_res

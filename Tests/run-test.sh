#!/bin/bash

set -ex

TEST=$1; shift

# 1 to test MPI
TEST_MPI=$1; shift
# 1 to test in CUDA mode
CUDA_MODE=$1; shift

ARGS=$@

CUR_DIR=`pwd`

# check exit code
function check_res ()
{
  if [ $? -ne 0 ]; then
    exit 1
  fi
}

if [ "$CUDA_MODE" == "1" ]; then
  echo "TESTING GPU"
  $CUR_DIR/Tests/suite/$TEST/build.sh 1 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh 1 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING GPU+PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh 3 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh 3 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
else
  echo "TESTING SEQUENTIAL"
  $CUR_DIR/Tests/suite/$TEST/build.sh 0 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh 0 $ARGS
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh 2 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh 2 $ARGS
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
fi

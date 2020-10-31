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
  $CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 1
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 1
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING GPU+PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 3
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 3
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
else
  echo "TESTING SEQUENTIAL"
  $CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 0
  check_res
  $CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 0
  check_res
  $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
  check_res

  if [ "$TEST_MPI" == "1" ]; then
    echo "TESTING PARALLEL"
    $CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 2
    check_res
    $CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 2
    check_res
    $CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
    check_res
  fi
fi

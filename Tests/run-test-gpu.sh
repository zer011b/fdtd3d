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

echo "TESTING GPU"
$CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 1
check_res
$CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 1
check_res
$CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
check_res

echo "TESTING GPU+MPI"
$CUR_DIR/Tests/suite/$TEST/build.sh $ARGS 3
check_res
$CUR_DIR/Tests/suite/$TEST/run.sh $ARGS 3
check_res
$CUR_DIR/Tests/suite/$TEST/cleanup.sh $ARGS
check_res

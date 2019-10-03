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
$CUR_DIR/Tools/TestSuite/suite/$TEST/build.sh $ARGS 1
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/run.sh $ARGS 1
check_res
$CUR_DIR/Tools/TestSuite/suite/$TEST/cleanup.sh $ARGS
check_res

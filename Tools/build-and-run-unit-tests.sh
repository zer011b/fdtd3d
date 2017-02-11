#!/bin/bash

# Home directory of project where root CMakeLists.txt is placed
HOME_DIR=$1

total_res=$((0))

BUILD_DIR=${HOME_DIR}/Build/UnitTestParallelGrid
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
touch ${BUILD_DIR}/build.log
${HOME_DIR}/Tools/Tests/build-and-run-unit-test-parallel-grid.sh ${HOME_DIR} ${BUILD_DIR} &> ${BUILD_DIR}/build.log

res=$(echo $?)

if [[ res -ne 0 ]];then
  echo "Unit test unit-test-parallel-grid failed"
  total_res=$((1))
else
  echo "Unit test unit-test-parallel-grid successful"
fi

if [[ total_res -ne 0 ]];then
  echo "Unit tests failed"
  exit 1
else
  echo "Unit tests successful"
fi

exit 0

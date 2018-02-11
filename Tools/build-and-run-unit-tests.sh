#!/bin/bash

# Home directory of project where root CMakeLists.txt is placed
HOME_DIR=$1

# CXX compiler
CXX_COMPILER=$2

# C compiler
C_COMPILER=$3

# true if Travis CI build
IS_TRAVIS_CI_BUILD=$4

total_res=$((0))

BUILD_DIR=${HOME_DIR}/Build

# ==== Parallel Grid unit test ====

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

if [[ $IS_TRAVIS_CI_BUILD -eq "true" ]]; then
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-parallel-grid.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER}
else
  touch ${BUILD_DIR}/build.log
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-parallel-grid.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER} &> ${BUILD_DIR}/build.log
fi

res=$(echo $?)

if [[ res -ne 0 ]]; then
  echo "Unit test unit-test-parallel-grid failed. See log at ${BUILD_DIR}/build.log"
  total_res=$((1))
else
  echo "Unit test unit-test-parallel-grid successful"
fi

# ==== Grid unit test ====

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

if [[ $IS_TRAVIS_CI_BUILD -eq "true" ]]; then
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-grid.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER}
else
  touch ${BUILD_DIR}/build.log
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-grid.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER} &> ${BUILD_DIR}/build.log
fi

res=$(echo $?)

if [[ res -ne 0 ]]; then
  echo "Unit test unit-test-grid failed. See log at ${BUILD_DIR}/build.log"
  total_res=$((1))
else
  echo "Unit test unit-test-grid successful"
fi

# ==== Dumpers/Loaders unit test ====

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

if [[ $IS_TRAVIS_CI_BUILD -eq "true" ]]; then
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-dumpers-loaders.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER}
else
  touch ${BUILD_DIR}/build.log
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-dumpers-loaders.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER} &> ${BUILD_DIR}/build.log
fi

res=$(echo $?)

if [[ res -ne 0 ]]; then
  echo "Unit test unit-test-dumpers-loaders failed. See log at ${BUILD_DIR}/build.log"
  total_res=$((1))
else
  echo "Unit test unit-test-dumpers-loaders successful"
fi

# ==== Coordinate unit test ====

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

if [[ $IS_TRAVIS_CI_BUILD -eq "true" ]]; then
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-coordinate.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER}
else
  touch ${BUILD_DIR}/build.log
  ${HOME_DIR}/Tools/Tests/build-and-run-unit-test-coordinate.sh ${HOME_DIR} ${BUILD_DIR} ${CXX_COMPILER} ${C_COMPILER} &> ${BUILD_DIR}/build.log
fi

res=$(echo $?)

if [[ res -ne 0 ]]; then
  echo "Unit test unit-test-coordinate failed. See log at ${BUILD_DIR}/build.log"
  total_res=$((1))
else
  echo "Unit test coordinate successful"
fi

# ==== ====

if [[ total_res -ne 0 ]]; then
  echo "Unit tests failed"
  exit 1
else
  echo "Unit tests successful"
fi

exit 0

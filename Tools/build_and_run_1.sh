#!/bin/bash

# Build and run test script.
# Parallel run is tested only on minimum number of nodes for this grid size.

SOURCE_PATH="${1}"
BUILD_TYPE="${2}"
PARALLEL_GRID_DIMENSION="${3}"
PARALLEL_GRID="${4}"
PARALLEL_BUFFER="${5}"
NUMBER_PROCESSES="${6}"

EXECUTABLE="fdtd3d"

BUILD_TEST_SCRIPT="${SOURCE_PATH}/tools/build_and_run.sh"

# ====================================================
# Script
# Source path
# Build type
# Values type
# Time steps number
# Grid dimension
# Print messages
# Parallel grid
# Parallel buffer dimensions
# Executable
# Number processes

count=$((0))

# ==== 1D non parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} float 0 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} float 1 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} float 2 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} double 0 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} double 1 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} double 2 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} long_double 0 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} long_double 1 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} ${BUILD_TYPE} long_double 2 ${PARALLEL_GRID_DIMENSION} OFF ${PARALLEL_GRID} ${PARALLEL_BUFFER} ${EXECUTABLE} ${NUMBER_PROCESSES}
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_and_run_1' ${count}"
  exit 1
fi

exit 0

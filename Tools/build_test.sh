#!/bin/bash

# Build and run test script.
# Parallel run is tested only on minimum number of nodes for this grid size.

SOURCE_PATH="${1}"

BUILD_TEST_SCRIPT="${SOURCE_PATH}/tools/build_and_run_1.sh"

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
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 1 OFF x 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

# ==== 1D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 1 ON x 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

# ==== 2D non parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 2 OFF x 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

# ==== 2D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 2 ON x 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 2 ON y 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 2 ON xy 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

# ==== 3D non parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 OFF x 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

# ==== 2D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON x 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON y 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON z 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON xy 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON yz 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON xz 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug 3 ON xyz 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed 'build_test' ${count}"
  exit 1
fi

echo "Build testing completed successfuly"
exit 0

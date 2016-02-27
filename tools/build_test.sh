#!/bin/bash

# Build and run test script.
# Parallel run is tested only on minimum number of nodes for this grid size.

SOURCE_PATH="${1}"
EXECUTABLE="fdtd3d"

BUILD_TEST_SCRIPT="${SOURCE_PATH}/tools/build_test.sh"

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
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 1 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

# ==== 1D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 1 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi


# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================


# ==== 2D non parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 2 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

# ==== 2D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 2 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 2 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 2 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi


# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ==================================================================================================


# ==== 3D non parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF OFF x ${EXECUTABLE} 1
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

# ==== 2D parallel ====
${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON x ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON y ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON z ${EXECUTABLE} 2
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON xy ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON yz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON xz ${EXECUTABLE} 4
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 0 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 1 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug float 2 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 0 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 1 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug double 2 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 0 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 1 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi

${BUILD_TEST_SCRIPT} ${SOURCE_PATH} Debug long_double 2 3 OFF ON xyz ${EXECUTABLE} 8
STATUS_CODE=$?
count=$((count+1))

if [ $STATUS_CODE -ne 0 ]; then
  echo "Failed ${count}"
  exit 1
fi



exit 0

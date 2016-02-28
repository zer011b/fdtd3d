#!/bin/bash

# Build script

SOURCE_PATH="${1}"
BUILD_DIR="${2}"
BUILD_TYPE="${3}"
VALUES_TYPE="${4}"
TIME_STEPS="${5}"
GRID_DIMENSION="${6}"
PRINT_MESSAGE="${7}"
PARALLEL_GRID="${8}"
PARALLEL_BUFFER="${9}"

if [ -d ${BUILD_DIR} ]; then
  rm -rf ${BUILD_DIR}
fi

mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

if [ "${VALUES_TYPE}" == "float" ]; then
  VALUES_TYPE="f"
fi
if [ "${VALUES_TYPE}" == "double" ]; then
  VALUES_TYPE="d"
fi
if [ "${VALUES_TYPE}" == "long_double" ]; then
  VALUES_TYPE="ld"
fi

cmake ${SOURCE_PATH} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DFULL_VALUES=${VALUES_TYPE} -DTIME_STEPS=${TIME_STEPS} -DGRID_DIMENSION=${GRID_DIMENSION} -DPRINT_MESSAGE=${PRINT_MESSAGE} -DPARALLEL_GRID=${PARALLEL_GRID} -DPARALLEL_BUFFER_DIMENSION=${PARALLEL_BUFFER}
STATUS_CODE=$?

if [ $STATUS_CODE -ne 0 ]; then
  echo "CMAKE failed"
  exit 1
fi

make
STATUS_CODE=$?

if [ $STATUS_CODE -ne 0 ]; then
  echo "MAKE failed"
  exit 1
fi

cd ..

echo "Build successful"
exit 0

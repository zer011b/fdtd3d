#!/bin/bash

# Run script

set -e

BUILD_DIR="${1}"
EXECUTABLE="${2}"
PARALLEL_RUN="${3}"
NUMBER_PROCESSES="${4}"

if [ ! -d ${BUILD_DIR} ]; then
  exit 1
fi

EXECUTABLE="${1}/${2}"

if [ "${PARALLEL_RUN}" == "ON" ]; then
  mpirun -n ${NUMBER_PROCESSES} ${EXECUTABLE}
else
  ${EXECUTABLE}
fi

STATUS_CODE=$?

if [ $STATUS_CODE -ne 0 ]; then
  echo "Run failed"
  exit 1
fi

echo "Run successful"
exit 0

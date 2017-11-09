#!/bin/bash

# Home directory of project where root CMakeLists.txt is placed
HOME_DIR=$1

# Build directory of unit test
BUILD_DIR=$2

# CXX compiler
CXX_COMPILER=$3

# C compiler
C_COMPILER=$4

CXX11_ENABLED=$5

COMPLEX_FIELD_VALUES=$6

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

function build
{
  PARALLEL_GRID_DIM=$1
  LIST_OF_BUFFERS="$2"

  for VALUE_TYPE in f d ld; do
    for TIME_STEPS in 1 2; do
      for PARALLEL_BUFFER in `echo $LIST_OF_BUFFERS`; do

        if [ "${VALUE_TYPE}" == "ld" ] && [ "${COMPLEX_FIELD_VALUES}" == "ON" ]; then
          continue
        fi

        cmake ${HOME_DIR} -DCMAKE_BUILD_TYPE=ReleaseWithAsserts \
          -DVALUE_TYPE=${VALUE_TYPE} \
          -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
          -DTIME_STEPS=${TIME_STEPS} \
          -DPARALLEL_GRID_DIMENSION=${PARALLEL_GRID_DIM} \
          -DPRINT_MESSAGE=ON \
          -DPARALLEL_GRID=ON \
          -DPARALLEL_BUFFER_DIMENSION=${PARALLEL_BUFFER} \
          -DCXX11_ENABLED=${CXX11_ENABLED} \
          -DCUDA_ENABLED=OFF \
          -DCUDA_ARCH_SM_TYPE=sm_50 \
          -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
          -DCMAKE_C_COMPILER=${C_COMPILER}

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        make unit-test-parallel-grid

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        if [[ "$PARALLEL_BUFFER" = "x" ]]; then
          mpirun -n 2 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "y" ]]; then
          mpirun -n 2 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "z" ]]; then
          mpirun -n 2 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "xy" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "yz" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "xz" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        fi

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi

        if [[ "$PARALLEL_BUFFER" = "x" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "y" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "z" ]]; then
          mpirun -n 4 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "xy" ]]; then
          mpirun -n 16 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "yz" ]]; then
          mpirun -n 16 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "xz" ]]; then
          mpirun -n 16 ./Tests/unit-test-parallel-grid
        elif [[ "$PARALLEL_BUFFER" = "xyz" ]]; then
          mpirun -n 8 ./Tests/unit-test-parallel-grid
        fi

        res=$(echo $?)

        if [[ res -ne 0 ]]; then
          exit 1
        fi
      done
    done
  done
}

array1D="x"
build 1 "$array1D"

array2D="x y xy"
build 2 "$array2D"

array3D="x y z xy yz xz xyz"
build 3 "$array3D"

exit 0

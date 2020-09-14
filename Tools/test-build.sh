#!/bin/bash

# Test builds of all combinations of requested parameters
#
# To test builds of all combinations:
#   ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "" "" ""
#
# To test builds of all combinations except for cuda:
#   ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "OFF,sm" "" ""
#
# To test builds of all sequential combinations:
#   ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" ""

set -ex

# Home directory of project where root CMakeLists.txt is located
HOME_DIR=$1; shift
if [[ "$HOME_DIR" == "" ]]; then
  echo "HOME_DIR is required"
  exit 1
fi

# Build directory of unit test
BUILD_DIR=$1; shift
if [[ "$BUILD_DIR" == "" ]]; then
  echo "BUILD_DIR is required"
  exit 1
fi

COMPILERS_VALUES="$1"; shift
if [[ "$COMPILERS_VALUES" == "" ]]; then
  COMPILERS_VALUES="gcc,g++"
fi

CMAKE_BUILD_TYPE_VALUES=$1; shift
if [[ "$CMAKE_BUILD_TYPE_VALUES" == "" ]]; then
  CMAKE_BUILD_TYPE_VALUES="Release RelWithDebInfo Debug"
fi

CXX11_ENABLED_VALUES=$1; shift
if [[ "$CXX11_ENABLED_VALUES" == "" ]]; then
  CXX11_ENABLED_VALUES="ON OFF"
fi

COMPLEX_FIELD_VALUES_VALUES="$1"; shift
if [[ "$COMPLEX_FIELD_VALUES_VALUES" == "" ]]; then
  COMPLEX_FIELD_VALUES_VALUES="ON OFF"
fi

PRINT_MESSAGE_VALUES="$1"; shift
if [[ "$PRINT_MESSAGE_VALUES" == "" ]]; then
  PRINT_MESSAGE_VALUES="ON OFF"
fi

VALUE_TYPE_VALUES="$1"; shift
if [[ "$VALUE_TYPE_VALUES" == "" ]]; then
  VALUE_TYPE_VALUES="f d ld"
fi

PARALLEL_MODE_VALUES="$1"; shift
if [[ "$PARALLEL_MODE_VALUES" == "" ]]; then
  PARALLEL_MODE_VALUES="OFF,1,x ON,1,x ON,2,x ON,2,y ON,2,xy ON,3,x ON,3,y ON,3,z ON,3,xy ON,3,yz ON,3,xz ON,3,xyz"
fi

CUDA_MODE_VALUES="$1"; shift
if [[ "$CUDA_MODE_VALUES" == "" ]]; then
  CUDA_MODE_VALUES="ON,sm_35 ON,sm_50 OFF,sm"
fi

SOLVER_DIM_MODES_VALUES="$1"; shift
if [[ "$SOLVER_DIM_MODES_VALUES" == "" ]]; then
  SOLVER_DIM_MODES_VALUES="DIM1 EX_HY EX_HZ EY_HX EY_HZ EZ_HX EZ_HY DIM2 TEX TEY TEZ TMX TMY TMZ DIM3 ALL"
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

for COMPILERS in $COMPILERS_VALUES; do
for CMAKE_BUILD_TYPE in $CMAKE_BUILD_TYPE_VALUES; do
for CXX11_ENABLED in $CXX11_ENABLED_VALUES; do
for COMPLEX_FIELD_VALUES in $COMPLEX_FIELD_VALUES_VALUES; do
for PRINT_MESSAGE in $PRINT_MESSAGE_VALUES; do
for VALUE_TYPE in $VALUE_TYPE_VALUES; do
for PARALLEL_MODE in $PARALLEL_MODE_VALUES; do
for CUDA_MODE in $CUDA_MODE_VALUES; do
for SOLVER_DIM_MODES in $SOLVER_DIM_MODES_VALUES; do

  PARALLEL_GRID=$(echo $PARALLEL_MODE | awk -F ',' '{print $1}')
  PARALLEL_GRID_DIMENSION=$(echo $PARALLEL_MODE | awk -F ',' '{print $2}')
  PARALLEL_BUFFER_DIMENSION=$(echo $PARALLEL_MODE | awk -F ',' '{print $3}')

  CUDA_ENABLED=$(echo $CUDA_MODE | awk -F ',' '{print $1}')
  CUDA_ARCH_SM_TYPE=$(echo $CUDA_MODE | awk -F ',' '{print $2}')

  CMAKE_C_COMPILER=$(echo $COMPILERS | awk -F ',' '{print $1}')
  CMAKE_CXX_COMPILER=$(echo $COMPILERS | awk -F ',' '{print $2}')

  # Exceptions
  if [ "${VALUE_TYPE}" == "ld" ] && [ "${CUDA_ENABLED}" == "ON" ]; then
    continue
  fi

  if [ "${CMAKE_BUILD_TYPE}" == "Debug" ] && [ "${CUDA_ENABLED}" == "ON" ] && [ "${SOLVER_DIM_MODES}" == "ALL" ]; then
    continue
  fi

  cmake ${HOME_DIR} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DVALUE_TYPE=${VALUE_TYPE} \
    -DCOMPLEX_FIELD_VALUES=${COMPLEX_FIELD_VALUES} \
    -DPARALLEL_GRID_DIMENSION=${PARALLEL_GRID_DIMENSION} \
    -DPRINT_MESSAGE=${PRINT_MESSAGE} \
    -DPARALLEL_GRID=${PARALLEL_GRID} \
    -DPARALLEL_BUFFER_DIMENSION=${PARALLEL_BUFFER_DIMENSION} \
    -DCXX11_ENABLED=${CXX11_ENABLED} \
    -DCUDA_ENABLED=${CUDA_ENABLED} \
    -DCUDA_ARCH_SM_TYPE=${CUDA_ARCH_SM_TYPE} \
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
    -DSOLVER_DIM_MODES=${SOLVER_DIM_MODES}

  # TODO: add next flags to testing:
  # -DDYNAMIC_GRID
  # -DMPI_DYNAMIC_CLOCK
  # -DCOMBINED_SENDRECV

  res=$(echo $?)
  if [[ res -ne 0 ]]; then
    echo "CMake failed"
    exit 2
  fi

  make fdtd3d

  res=$(echo $?)
  if [[ res -ne 0 ]]; then
    echo "Make failed"
    exit 2
  fi

  echo "Build successful"
done
done
done
done
done
done
done
done
done

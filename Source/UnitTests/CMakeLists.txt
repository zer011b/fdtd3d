#
#  Copyright (C) 2017 Gleb Balykov
#
#  This file is part of fdtd3d.
#
#  fdtd3d is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  fdtd3d is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with fdtd3d; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

include_directories ("${PROJECT_SOURCE_DIR}/Source/Helpers")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Physics")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Kernels")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Settings/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Coordinate")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Grid")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Layout")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Dumper")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Scheme")

# Tests that link against HIP libraries must be HIP executables to handle device-link.
# However, some tests have #error directives that require certain build options (e.g.,
# all DIM modes enabled). We only mark tests as HIP that we know will compile with the
# current build configuration. The main fdtd3d executable and GPU tests are the priority.
if ("${USE_HIP}")
  # Mark C++ source files as HIP language for tests that link HIP libraries
  set_source_files_properties(unit-test-settings.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(unit-test-clock.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(unit-test-grid.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(unit-test-coordinate.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(unit-test-layout.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(unit-test-approximation.cpp PROPERTIES LANGUAGE HIP)
  # unit-test-dumpers-loaders requires all DIM modes to be enabled; skip HIP marking
  # to avoid compile errors when building with a subset of modes
endif()

add_executable (unit-test-settings unit-test-settings.cpp)
target_link_libraries (unit-test-settings Settings Kernels)

add_executable (unit-test-clock unit-test-clock.cpp)
target_link_libraries (unit-test-clock Settings Kernels ${MPI_CXX_LIBRARIES})

# These tests require PARALLEL_GRID and MPI
if ("${PARALLEL_GRID}")
  add_executable (unit-test-parallel-grid unit-test-parallel-grid.cpp)
  target_link_libraries (unit-test-parallel-grid Grid Layout)

  add_executable (perf-test-parallel-grid perf-test-parallel-grid.cpp)
  target_link_libraries (perf-test-parallel-grid Grid Layout)
endif()

add_executable (unit-test-grid unit-test-grid.cpp)
target_link_libraries (unit-test-grid Coordinate Grid)

# This test requires all DIM modes. When building with HIP and a subset of modes,
# it cannot compile due to #error directives, and also cannot link due to HIP device
# code in libraries. Only build it when all modes are enabled or HIP is not used.
if (NOT "${USE_HIP}" OR "${SOLVER_DIM_MODES}" STREQUAL "ALL")
  add_executable (unit-test-dumpers-loaders unit-test-dumpers-loaders.cpp)
  target_link_libraries (unit-test-dumpers-loaders Coordinate Grid Loader Dumper)
  if ("${USE_HIP}" AND "${SOLVER_DIM_MODES}" STREQUAL "ALL")
    set_source_files_properties(unit-test-dumpers-loaders.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(unit-test-dumpers-loaders PROPERTIES
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      HIP_SEPARABLE_COMPILATION ON)
  endif()
endif()

add_executable (unit-test-coordinate unit-test-coordinate.cpp)
target_link_libraries (unit-test-coordinate Settings Coordinate)

add_executable (unit-test-layout unit-test-layout.cpp)
target_link_libraries (unit-test-layout Settings Layout)

add_executable (unit-test-approximation unit-test-approximation.cpp)
target_link_libraries (unit-test-approximation Settings Layout)

add_executable (unit-test-complex unit-test-complex.cpp)
target_link_libraries (unit-test-complex Helpers)

if ("${USE_HIP}")
  # Set HIP properties for tests that link HIP libraries
  set_target_properties(unit-test-settings PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(unit-test-clock PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(unit-test-grid PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(unit-test-coordinate PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(unit-test-layout PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  set_target_properties(unit-test-approximation PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
endif()

if ("${USE_HIP}")
  set_source_files_properties (unit-test-cuda-grid.cu PROPERTIES LANGUAGE HIP)
  add_executable (unit-test-cuda-grid unit-test-cuda-grid.cu)
  set_target_properties (unit-test-cuda-grid PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
  target_link_libraries (unit-test-cuda-grid PRIVATE Grid Coordinate)

  set_source_files_properties (unit-test-internalscheme.cu PROPERTIES LANGUAGE HIP)
  add_executable (unit-test-internalscheme unit-test-internalscheme.cu)
  set_target_properties (unit-test-internalscheme PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_SEPARABLE_COMPILATION ON)
elseif ("${CUDA_ENABLED}")
  add_executable (unit-test-cuda-grid unit-test-cuda-grid.cu)
  set_target_properties (unit-test-cuda-grid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries (unit-test-cuda-grid PRIVATE Grid Coordinate)

  add_executable (unit-test-internalscheme unit-test-internalscheme.cu)
  set_target_properties (unit-test-internalscheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_executable (unit-test-internalscheme unit-test-internalscheme.cpp)
endif ()

target_link_libraries (unit-test-internalscheme InternalScheme)

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

add_executable (unit-test-settings unit-test-settings.cpp)
target_link_libraries (unit-test-settings Settings Kernels)

add_executable (unit-test-clock unit-test-clock.cpp)
target_link_libraries (unit-test-clock Settings Kernels ${MPI_CXX_LIBRARIES})

add_executable (unit-test-parallel-grid unit-test-parallel-grid.cpp)
target_link_libraries (unit-test-parallel-grid Grid Layout)

add_executable (perf-test-parallel-grid perf-test-parallel-grid.cpp)
target_link_libraries (perf-test-parallel-grid Grid Layout)

add_executable (unit-test-grid unit-test-grid.cpp)
target_link_libraries (unit-test-grid Coordinate Grid)

add_executable (unit-test-dumpers-loaders unit-test-dumpers-loaders.cpp)
target_link_libraries (unit-test-dumpers-loaders Coordinate Grid Loader Dumper)

add_executable (unit-test-coordinate unit-test-coordinate.cpp)
target_link_libraries (unit-test-coordinate Settings Coordinate)

add_executable (unit-test-layout unit-test-layout.cpp)
target_link_libraries (unit-test-layout Settings Layout)

add_executable (unit-test-approximation unit-test-approximation.cpp)
target_link_libraries (unit-test-approximation Settings Layout)

add_executable (unit-test-complex unit-test-complex.cpp)
target_link_libraries (unit-test-complex Helpers)

if ("${CUDA_ENABLED}")
  add_executable (unit-test-cuda-grid unit-test-cuda-grid.cu)
  set_target_properties (unit-test-cuda-grid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries (unit-test-cuda-grid PRIVATE Grid Coordinate)

  add_executable (unit-test-internalscheme unit-test-internalscheme.cu)
  set_target_properties (unit-test-internalscheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_executable (unit-test-internalscheme unit-test-internalscheme.cpp)
endif ()

target_link_libraries (unit-test-internalscheme InternalScheme)

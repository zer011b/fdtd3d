#
#  Copyright (C) 2015 Gleb Balykov
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
add_subdirectory (Helpers)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Physics")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Kernels")
add_subdirectory (Kernels)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Settings/")
add_subdirectory (Settings)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Coordinate")
add_subdirectory (Coordinate)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Grid")
add_subdirectory (Grid)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Layout")
add_subdirectory (Layout)

include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Dumper")
add_subdirectory (File-Management)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Scheme")
add_subdirectory (Scheme)

add_subdirectory (Tools)

add_subdirectory (UnitTests)

if ("${CUDA_ENABLED}")
  add_executable (fdtd3d main.cu)
  set_target_properties (fdtd3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_executable (fdtd3d main.cpp)
  add_executable (fdtd3dbench main.cpp)
  target_compile_definitions (fdtd3dbench PUBLIC PREDEFINED_ARGS)
  target_link_libraries (fdtd3dbench Scheme)
endif ()

target_link_libraries (fdtd3d Scheme)

#set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
#set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

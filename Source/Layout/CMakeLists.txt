#
#  Copyright (C) 2016 Gleb Balykov
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

file(GLOB_RECURSE LAYOUT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_LAYOUT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if ("${CUDA_ENABLED}")
  list(FILTER LAYOUT_SOURCES EXCLUDE REGEX ".*/YeeGridLayout.cpp$")
  list(FILTER LAYOUT_SOURCES EXCLUDE REGEX ".*/Approximation.cpp$")
  add_library (Layout ${LAYOUT_SOURCES} ${CUDA_LAYOUT_SOURCES})
  set_target_properties (Layout PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (Layout ${LAYOUT_SOURCES})
endif ()

target_link_libraries (Layout Coordinate Kernels)

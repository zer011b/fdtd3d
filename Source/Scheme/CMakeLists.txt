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

file(GLOB_RECURSE INTERNAL_SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InternalScheme.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CallBack.cpp")
file(GLOB_RECURSE CUDA_INTERNAL_SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InternalScheme.cu" "${CMAKE_CURRENT_SOURCE_DIR}/CallBack.cu")

file(GLOB_RECURSE SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Scheme*.cpp")
file(GLOB_RECURSE CUDA_SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Scheme*.cu")

if ("${CUDA_ENABLED}")
  list(FILTER INTERNAL_SCHEME_SOURCES EXCLUDE REGEX ".*/InternalScheme*.cpp$")
  list(FILTER INTERNAL_SCHEME_SOURCES EXCLUDE REGEX ".*/CallBack.cpp$")
  add_library (InternalScheme ${INTERNAL_SCHEME_SOURCES} ${CUDA_INTERNAL_SCHEME_SOURCES})
  set_target_properties (InternalScheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  list(FILTER SCHEME_SOURCES EXCLUDE REGEX ".*/Scheme*.cpp$")
  add_library (Scheme ${SCHEME_SOURCES} ${CUDA_SCHEME_SOURCES})
  set_target_properties (Scheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (InternalScheme ${INTERNAL_SCHEME_SOURCES})
  add_library (Scheme ${SCHEME_SOURCES})
endif ()

target_link_libraries (InternalScheme Layout Settings Grid)
target_link_libraries (Scheme InternalScheme Dumper Loader)

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

file(GLOB_RECURSE SETTINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_SETTINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if ("${CUDA_ENABLED}")
  list(FILTER SETTINGS_SOURCES EXCLUDE REGEX ".*/Settings.cpp$")
  add_library (Settings ${SETTINGS_SOURCES} ${CUDA_SETTINGS_SOURCES})
  set_target_properties (Settings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (Settings ${SETTINGS_SOURCES})
endif ()

target_link_libraries (Settings Helpers)

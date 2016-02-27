cmake_minimum_required (VERSION 3.0.2)

option(PRINT_MESSAGE "Print all messages" OFF)
option(PARALLEL_GRID "Use parallel grid" OFF)

set(FULL_VALUES "d" CACHE STRING "Defines type of values")
set(TIME_STEPS "2" CACHE STRING "Defines number of time steps used")
set(GRID_DIMENSION "2" CACHE STRING "Defines number dimension in grid")
set(PARALLEL_BUFFER_DIMENSION "x" CACHE STRING "Defines dimension of parallel grid")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set (CMAKE_BUILD_TYPE "Debug")
endif ()

if ("${CMAKE_CXX_COMPILER}" STREQUAL "")
  set (CMAKE_CXX_COMPILER "mpicxx")
endif ()

if ("${CMAKE_C_COMPILER}" STREQUAL "")
  set (CMAKE_C_COMPILER "mpicc")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message ("Build type: ${CMAKE_BUILD_TYPE}.")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message ("Build type: ${CMAKE_BUILD_TYPE}.")
endif ()

message ("C++ compiler: ${CMAKE_CXX_COMPILER}.")
message ("C compiler: ${CMAKE_C_COMPILER}.")

if ("${FULL_VALUES}" STREQUAL "f")
  message ("Values: float.")
  add_definitions (-DFLOAT_VALUES="")
elseif ("${FULL_VALUES}" STREQUAL "d")
  message ("Values: double.")
  add_definitions (-DDOUBLE_VALUES="")
elseif ("${FULL_VALUES}" STREQUAL "ld")
  message ("Values: long double.")
  add_definitions (-DLONG_DOUBLE_VALUES="")
else ()
  message(FATAL_ERROR "Unknown values type")
endif ()

if ("${PRINT_MESSAGE}")
  message ("Print messages.")
  add_definitions (-DPRINT_MESSAGE=1)
else ()
  message ("Do not print messages.")
  add_definitions (-DPRINT_MESSAGE=0)
endif ()

if ("${TIME_STEPS}" STREQUAL "1")
  add_definitions (-DONE_TIME_STEP="")
elseif ("${TIME_STEPS}" STREQUAL "2")
  add_definitions (-DTWO_TIME_STEPS="")
else ()
  #message(FATAL_ERROR "Unknown TIME_STEPS")
  add_definitions (-DTWO_TIME_STEPS="")
endif ()

message ("Time steps: ${TIME_STEPS}.")

if ("${GRID_DIMENSION}" STREQUAL "1")
  add_definitions (-DGRID_1D="")
elseif ("${GRID_DIMENSION}" STREQUAL "2")
  add_definitions (-DGRID_2D="")
elseif ("${GRID_DIMENSION}" STREQUAL "3")
  add_definitions (-DGRID_3D="")
else ()
  #message(FATAL_ERROR "Unknown GRID_DIMENSION")
  add_definitions (-DGRID_2D="")
endif ()

message ("Grid dimension: ${GRID_DIMENSION}.")

if ("${PARALLEL_GRID}")
  message ("Parallel grid: ON.")
  add_definitions (-DPARALLEL_GRID="")

  if ("${GRID_DIMENSION}" STREQUAL "1")

    if ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "x")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_X="")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "y")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "z")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "yz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xyz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    else ()
      message(FATAL_ERROR "Unknown PARALLEL_BUFFER_DIMENSION")
    endif ()

  elseif ("${GRID_DIMENSION}" STREQUAL "2")

    if ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "x")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_X="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "y")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_Y="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "z")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_2D_XY="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "yz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xyz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this GRID_DIMENSION")
    else ()
      message(FATAL_ERROR "Unknown PARALLEL_BUFFER_DIMENSION")
    endif ()

  elseif ("${GRID_DIMENSION}" STREQUAL "3")

    if ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "x")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_X="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "y")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_Y="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "z")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_Z="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_2D_XY="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "yz")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_2D_YZ="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xz")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_2D_XZ="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xyz")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_3D_XYZ="")
    else ()
      message(FATAL_ERROR "Unknown PARALLEL_BUFFER_DIMENSION")
    endif ()

  else ()
    message(FATAL_ERROR "Unknown GRID_DIMENSION")
  endif ()
else ()
  message ("Parallel grid: OFF.")
endif ()

add_definitions (-std=c++11)


set (BUILD_FLAGS "")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set (BUILD_FLAGS "${BUILD_FLAGS} -flto -fno-fat-lto-objects")
endif ()

#set (BUILD_FLAGS "${BUILD_FLAGS} -fno-stack-protector")
#set (BUILD_FLAGS "${BUILD_FLAGS} -fno-exceptions -fno-rtti")
#set (BUILD_FLAGS "${BUILD_FLAGS} -Wall -Wextra -pedantic")
#set (BUILD_FLAGS "${BUILD_FLAGS} -Wformat-nonliteral -Winit-self -Wno-stack-protector")
#set (BUILD_FLAGS "${BUILD_FLAGS} -Wconversion -Wsign-conversion -Wformat-security")
#set (BUILD_FLAGS "${BUILD_FLAGS} -Wmissing-declarations -Wno-attributes")
#set (BUILD_FLAGS "${BUILD_FLAGS} -Wfatal-errors")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set (BUILD_FLAGS "${BUILD_FLAGS} -O2")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set (BUILD_FLAGS "${BUILD_FLAGS} -O0")
endif ()

set (COMPILE_FLAGS "${COMPILE_FLAGS} ${BUILD_FLAGS}")


find_package (MPI REQUIRED)
message ("${MPI_INCLUDE_PATH}")

###### Project fdtd ######
project (fdtd3d)
message ("Project dir: ${PROJECT_SOURCE_DIR}")

# Include MPI
include_directories (${MPI_INCLUDE_PATH})
set (LIBS ${LIBS} ${MPI_LIBRARIES})
if (MPI_COMPILE_FLAGS)
  set (COMPILE_FLAGS ${COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
endif ()

if (MPI_LINK_FLAGS)
  set (LINK_FLAGS ${LINK_FLAGS} ${MPI_LINK_FLAGS})
endif ()

message ("Compile flags: ${COMPILE_FLAGS}")
message ("Link flags: ${LINK_FLAGS}")


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/third-party/EasyBMP/source")
  execute_process (COMMAND wget http://prdownloads.sourceforge.net/easybmp/EasyBMP_1.06.zip)
  execute_process (COMMAND mkdir ${PROJECT_SOURCE_DIR}/third-party/EasyBMP/source)
  execute_process (COMMAND unzip EasyBMP_1.06.zip -d ${PROJECT_SOURCE_DIR}/third-party/EasyBMP/source)
  execute_process (COMMAND rm EasyBMP_1.06.zip)
endif ()

# Add EasyBMP library
# Include headers of EasyBMP
include_directories ("${PROJECT_SOURCE_DIR}/third-party/EasyBMP/source")
# Include CMakeLists.txt from the subfolder
add_subdirectory (${PROJECT_SOURCE_DIR}/third-party/EasyBMP)

include_directories ("${PROJECT_SOURCE_DIR}/Helpers")
add_subdirectory (Helpers)

# Add Grid library
# Include headers of Grid
include_directories ("${PROJECT_SOURCE_DIR}/Grid")
# Include CMakeLists.txt from the subfolder
add_subdirectory (Grid)
set (LIBS ${LIBS} Grid)


# Add File management library
# Include headers of Loader and Dumper
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/Dumper")
# Include CMakeLists.txt from the subfolder
add_subdirectory (File-Management)
set (LIBS ${LIBS} Loader)
set (LIBS ${LIBS} Dumper)


include_directories ("${PROJECT_SOURCE_DIR}/Scheme")
add_subdirectory (Scheme)
set (LIBS ${LIBS} Scheme)


include_directories ("${PROJECT_SOURCE_DIR}/Solver")
add_subdirectory (Solver)
set (LIBS ${LIBS} Solver)


add_executable (fdtd3d main.cpp)
target_link_libraries (fdtd3d ${LIBS})

set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

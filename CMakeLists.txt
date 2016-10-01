cmake_minimum_required (VERSION 3.0.2)

project (fdtd3d)

option(PRINT_MESSAGE "Print all messages" OFF)
option(PARALLEL_GRID "Use parallel grid" OFF)
option(CUDA_ENABLED "Cuda support enabled" OFF)
option(CXX11_ENABLED "C++11 support enabled" OFF)

set(VALUE_TYPE "d" CACHE STRING "Defines type of values")
set(TIME_STEPS "2" CACHE STRING "Defines number of time steps used")
set(PARALLEL_GRID_DIMENSION "2" CACHE STRING "Defines number dimension in grid")
set(PARALLEL_BUFFER_DIMENSION "x" CACHE STRING "Defines dimension of parallel grid")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

if ("${CUDA_ENABLED}")
  find_package(CUDA QUIET REQUIRED)
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set (CMAKE_BUILD_TYPE "Debug")
endif ()

if ("${PARALLEL_GRID}")
  if ("${CMAKE_CXX_COMPILER}" STREQUAL "")
    set (CMAKE_CXX_COMPILER "mpicxx")
  endif ()

  if ("${CMAKE_C_COMPILER}" STREQUAL "")
    set (CMAKE_C_COMPILER "mpicc")
  endif ()
else ()
  if ("${CMAKE_CXX_COMPILER}" STREQUAL "")
    set (CMAKE_CXX_COMPILER "g++")
  endif ()

  if ("${CMAKE_C_COMPILER}" STREQUAL "")
    set (CMAKE_C_COMPILER "gcc")
  endif ()
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message ("Build type: ${CMAKE_BUILD_TYPE}.")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message ("Build type: ${CMAKE_BUILD_TYPE}.")
endif ()

message ("C++ compiler: ${CMAKE_CXX_COMPILER}.")
message ("C compiler: ${CMAKE_C_COMPILER}.")

if ("${VALUE_TYPE}" STREQUAL "f")
  message ("Values: float.")
  add_definitions (-DFLOAT_VALUES="")
elseif ("${VALUE_TYPE}" STREQUAL "d")
  message ("Values: double.")
  add_definitions (-DDOUBLE_VALUES="")
elseif ("${VALUE_TYPE}" STREQUAL "ld")
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

if ("${PARALLEL_GRID_DIMENSION}" STREQUAL "1")
  add_definitions (-DGRID_1D="")
elseif ("${PARALLEL_GRID_DIMENSION}" STREQUAL "2")
  add_definitions (-DGRID_2D="")
elseif ("${PARALLEL_GRID_DIMENSION}" STREQUAL "3")
  add_definitions (-DGRID_3D="")
else ()
  #message(FATAL_ERROR "Unknown PARALLEL_GRID_DIMENSION")
  add_definitions (-DGRID_2D="")
endif ()

message ("Grid dimension: ${PARALLEL_GRID_DIMENSION}.")

if ("${PARALLEL_GRID}")
  message ("Parallel grid: ON.")
  add_definitions (-DPARALLEL_GRID="")

  if ("${PARALLEL_GRID_DIMENSION}" STREQUAL "1")

    if ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "x")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_X="")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "y")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "z")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "yz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xyz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    else ()
      message(FATAL_ERROR "Unknown PARALLEL_BUFFER_DIMENSION")
    endif ()

  elseif ("${PARALLEL_GRID_DIMENSION}" STREQUAL "2")

    if ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "x")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_X="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "y")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_1D_Y="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "z")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xy")
      message ("Parallel dimension: ${PARALLEL_BUFFER_DIMENSION}.")
      add_definitions (-DPARALLEL_BUFFER_DIMENSION_2D_XY="")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "yz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    elseif ("${PARALLEL_BUFFER_DIMENSION}" STREQUAL "xyz")
      message(FATAL_ERROR "Unsupported PARALLEL_BUFFER_DIMENSION for this PARALLEL_GRID_DIMENSION")
    else ()
      message(FATAL_ERROR "Unknown PARALLEL_BUFFER_DIMENSION")
    endif ()

  elseif ("${PARALLEL_GRID_DIMENSION}" STREQUAL "3")

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
    message(FATAL_ERROR "Unknown PARALLEL_GRID_DIMENSION")
  endif ()
else ()
  message ("Parallel grid: OFF.")
endif ()

if ("${CUDA_ENABLED}")
  message ("Cuda: ON.")
  add_definitions (-DCUDA_ENABLED="")
endif ()

set (BUILD_FLAGS "")

if ("${CXX11_ENABLED}")
  add_definitions (-DCXX11_ENABLED)
  set (BUILD_FLAGS "${BUILD_FLAGS} -std=c++11")
endif ()

#if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#  set (BUILD_FLAGS "${BUILD_FLAGS} -flto -fno-fat-lto-objects")
#endif ()

#set (BUILD_FLAGS "${BUILD_FLAGS} -fno-stack-protector")
#set (BUILD_FLAGS "${BUILD_FLAGS} -fno-exceptions -fno-rtti")

#-pedantic -pedantic-errors

set (WARNING_FLAGS "")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Wfatal-errors")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Wall -Wextra")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Wformat=2 -Wformat-security -Wuninitialized")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Winit-self -Wno-stack-protector")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Wconversion -Wsign-conversion")
#set (WARNING_FLAGS "${WARNING_FLAGS} -Wmissing-declarations")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set (BUILD_FLAGS "${BUILD_FLAGS} -O2")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set (BUILD_FLAGS "${BUILD_FLAGS} -O0")
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS} ${WARNING_FLAGS}")

###### Project fdtd ######
message ("Project dir: ${PROJECT_SOURCE_DIR}")

# Include MPI
if ("${PARALLEL_GRID}")
  find_package (MPI REQUIRED)
  message ("${MPI_INCLUDE_PATH}")

  include_directories (${MPI_INCLUDE_PATH})
  set (LIBS ${LIBS} ${MPI_LIBRARIES})
  if (MPI_COMPILE_FLAGS)
    set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS})
  endif ()

  #if (MPI_LINK_FLAGS)
  #  set (LINK_FLAGS ${LINK_FLAGS} ${MPI_LINK_FLAGS})
  #endif ()
endif()

message ("Compile flags: ${CMAKE_CXX_FLAGS}")
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

include_directories ("${PROJECT_SOURCE_DIR}/Kernels")
add_subdirectory (Kernels)

if ("${CUDA_ENABLED}")
  set (CUDA_SOURCES "")
  include_directories ("${PROJECT_SOURCE_DIR}/Cuda")
  add_subdirectory (Cuda)
endif ()

# Add Grid library
# Include headers of Grid
include_directories ("${PROJECT_SOURCE_DIR}/Grid")
# Include CMakeLists.txt from the subfolder
add_subdirectory (Grid)
set (LIBS ${LIBS} Grid)

include_directories ("${PROJECT_SOURCE_DIR}/Physics")
add_subdirectory (Physics)
set (LIBS ${LIBS} Physics)


# Add File management library
# Include headers of Loader and Dumper
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/File-Management/Dumper")
# Include CMakeLists.txt from the subfolder
add_subdirectory (File-Management)
set (LIBS ${LIBS} Loader)
set (LIBS ${LIBS} Dumper)


include_directories("${PROJECT_SOURCE_DIR}")
include_directories ("${PROJECT_SOURCE_DIR}/Scheme")
add_subdirectory (Scheme)
set (LIBS ${LIBS} Scheme)

if ("${CUDA_ENABLED}")
  set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_50 -ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -m64")

  cuda_add_executable(fdtd3d main.cpp ${CUDA_SOURCES} ${PROJECT_SOURCE_DIR}/Kernels/Kernels.cpp)
else ()
  add_executable (fdtd3d main.cpp)
endif ()

target_link_libraries (fdtd3d ${LIBS})

#set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
#set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

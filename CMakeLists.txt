cmake_minimum_required (VERSION 3.0.2)

option(FULL_VALUES "Use double values" ON)
option(PRINT_MESSAGE "Print all messages" OFF)

set(TIME_STEPS "2" CACHE STRING "Defines number of time steps used")
set(GRID_DIMENSION "2" CACHE STRING "Defines number dimension in grid")

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

if ("${FULL_VALUES}")
  message ("Values: double.")
  add_definitions (-DFULL_VALUES=1)
else ()
  message ("Values: float.")
  add_definitions (-DFULL_VALUES=0)
endif ()

if ("${PRINT_MESSAGE}")
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message ("Print messages.")
    add_definitions (-DPRINT_MESSAGE=1)
  else ()
    message ("Do not print messages.")
    add_definitions (-DPRINT_MESSAGE=0)
  endif ()
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

add_definitions (-DPARALLEL_GRID="")
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

# Add Field library
# Include headers of Field
include_directories ("${PROJECT_SOURCE_DIR}/Field")
# Include CMakeLists.txt from the subfolder
add_subdirectory (Field)
set (LIBS ${LIBS} Field)


# Add Field library
# Include headers of Loader and Dumper
include_directories ("${PROJECT_SOURCE_DIR}/Grid-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Grid-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Grid-Management/Dumper")
# Include CMakeLists.txt from the subfolder
add_subdirectory (Grid-Management)
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

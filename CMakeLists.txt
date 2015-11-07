cmake_minimum_required (VERSION 3.0.2)

set (CMAKE_CXX_COMPILER "mpicxx")
set (CMAKE_C_COMPILER "mpicc")

set (COMPILE_FLAGS "-O3")
set (LINK_FLAGS "")

#add_definitions (-DSMALL_VALUES="")
add_definitions (-DFULL_VALUES="")
add_definitions (-DTWO_TIME_STEPS="")
add_definitions (-DGRID_2D="")
add_definitions (-std=c++11)

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


add_executable (fdtd3d main.cpp)
target_link_libraries (fdtd3d ${LIBS})

set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

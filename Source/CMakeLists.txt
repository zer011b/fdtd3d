message ("Source build.")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Helpers")
add_subdirectory (Helpers)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Physics")
add_subdirectory (Physics)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Kernels")
add_subdirectory (Kernels)
set (LIBS ${LIBS} Kernels)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Settings/")
add_subdirectory (Settings)
set (LIBS ${LIBS} Settings)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Coordinate")
add_subdirectory (Coordinate)
set (LIBS ${LIBS} Coordinate)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Grid")
add_subdirectory (Grid)
set (LIBS ${LIBS} Grid)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Layout")
add_subdirectory (Layout)
set (LIBS ${LIBS} Layout)

if ("${CUDA_ENABLED}")
  include_directories ("${PROJECT_SOURCE_DIR}/Source/Cuda")
  add_subdirectory (Cuda)
  set (LIBS ${LIBS} CudaScheme)
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Dumper")
add_subdirectory (File-Management)
set (LIBS ${LIBS} Loader)
set (LIBS ${LIBS} Dumper)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Scheme")
add_subdirectory (Scheme)
set (LIBS ${LIBS} Scheme)

add_executable (fdtd3d main.cpp)

target_link_libraries (fdtd3d ${LIBS})

#set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
#set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

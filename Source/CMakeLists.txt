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

if ("${CUDA_ENABLED}")
  add_executable (fdtd3d main.cu)
  set_target_properties (fdtd3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_executable (fdtd3d main.cpp)
endif ()

target_link_libraries (fdtd3d Scheme)

#set_target_properties (fdtd3d PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
#set_target_properties (fdtd3d PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

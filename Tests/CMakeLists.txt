message ("Source build.")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Helpers")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Physics")

include_directories ("${PROJECT_SOURCE_DIR}/Source/Kernels")
set (LIBS ${LIBS} Kernels)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Settings/")
set (LIBS ${LIBS} Settings)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Coordinate")
set (LIBS ${LIBS} Coordinate)

include_directories ("${PROJECT_SOURCE_DIR}/Source/Grid")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Layout")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Loader")
include_directories ("${PROJECT_SOURCE_DIR}/Source/File-Management/Dumper")
include_directories ("${PROJECT_SOURCE_DIR}/Source/Scheme")

add_executable (unit-test-settings unit-test-settings.cpp)
target_link_libraries (unit-test-settings Helpers Settings Kernels)

add_executable (unit-test-clock unit-test-clock.cpp)
target_link_libraries (unit-test-clock Helpers Settings Kernels ${MPI_CXX_LINK_FLAGS})

add_executable (unit-test-parallel-grid unit-test-parallel-grid.cpp)
target_link_libraries (unit-test-parallel-grid ${LIBS} Grid Helpers Settings Layout)

add_executable (perf-test-parallel-grid perf-test-parallel-grid.cpp)
target_link_libraries (perf-test-parallel-grid ${LIBS} Grid Helpers Settings Layout)

add_executable (unit-test-grid unit-test-grid.cpp)
target_link_libraries (unit-test-grid ${LIBS} Grid Helpers Settings)

add_executable (unit-test-dumpers-loaders unit-test-dumpers-loaders.cpp)
target_link_libraries (unit-test-dumpers-loaders ${LIBS} Grid Loader Dumper Helpers Settings)

add_executable (unit-test-coordinate unit-test-coordinate.cpp)
target_link_libraries (unit-test-coordinate Helpers Settings Coordinate)

add_executable (unit-test-layout unit-test-layout.cpp)
target_link_libraries (unit-test-layout Helpers Settings Coordinate Layout)

add_executable (unit-test-approximation unit-test-approximation.cpp)
target_link_libraries (unit-test-approximation Helpers Settings Layout)

add_executable (unit-test-complex unit-test-complex.cpp)
target_link_libraries (unit-test-complex Helpers)

if ("${CUDA_ENABLED}")
  add_executable (unit-test-cuda-grid unit-test-cuda-grid.cu)
  set_target_properties (unit-test-cuda-grid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries (unit-test-cuda-grid PRIVATE Grid Helpers Settings Kernels Coordinate)

if ("${CXX11_ENABLED}")
  target_compile_features (unit-test-cuda-grid PUBLIC cxx_std_11)
endif ()

  add_executable (unit-test-internalscheme unit-test-internalscheme.cu)
  set_target_properties (unit-test-internalscheme PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_executable (unit-test-internalscheme unit-test-internalscheme.cpp)
endif ()

target_link_libraries (unit-test-internalscheme ${LIBS} Grid Helpers Settings InternalScheme)

if ("${CXX11_ENABLED}")
  target_compile_features (unit-test-settings PUBLIC cxx_std_11)
  target_compile_features (unit-test-clock PUBLIC cxx_std_11)
  target_compile_features (unit-test-parallel-grid PUBLIC cxx_std_11)
  target_compile_features (perf-test-parallel-grid PUBLIC cxx_std_11)
  target_compile_features (unit-test-grid PUBLIC cxx_std_11)
  target_compile_features (unit-test-dumpers-loaders PUBLIC cxx_std_11)
  target_compile_features (unit-test-coordinate PUBLIC cxx_std_11)
  target_compile_features (unit-test-layout PUBLIC cxx_std_11)
  target_compile_features (unit-test-approximation PUBLIC cxx_std_11)
  target_compile_features (unit-test-complex PUBLIC cxx_std_11)
  target_compile_features (unit-test-internalscheme PUBLIC cxx_std_11)
endif ()

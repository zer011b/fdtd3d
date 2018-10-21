message ("Layout build.")

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

if ("${CXX11_ENABLED}")
  target_compile_features (Layout PUBLIC cxx_std_11)
endif ()

target_link_libraries (Layout Coordinate Kernels)

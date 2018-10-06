message ("Kernels build.")

file(GLOB_RECURSE KERNELS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_KERNELS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if ("${CUDA_ENABLED}")
  list(FILTER KERNELS_SOURCES EXCLUDE REGEX ".*/FieldValue.cpp$")
  add_library (Kernels ${KERNELS_SOURCES} ${CUDA_KERNELS_SOURCES})
  set_target_properties (Kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (Kernels ${KERNELS_SOURCES})
endif ()

if ("${CXX11_ENABLED}")
  target_compile_features (Kernels PUBLIC cxx_std_11)
endif ()

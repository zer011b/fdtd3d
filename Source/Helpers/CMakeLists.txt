message ("Helpers build.")

file(GLOB_RECURSE HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if ("${CUDA_ENABLED}")
  add_library (Helpers ${HELPER_SOURCES} ${CUDA_HELPER_SOURCES})
  set_target_properties (Helpers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  add_library (Helpers ${HELPER_SOURCES})
endif ()

if ("${CXX11_ENABLED}")
  target_compile_features (Helpers PUBLIC cxx_std_11)
endif ()

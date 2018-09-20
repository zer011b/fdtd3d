message ("Coordinate build.")

file(GLOB_RECURSE COORDINATE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Coordinate ${COORDINATE_SOURCES})

if ("${CXX11_ENABLED}")
  target_compile_features (Coordinate PUBLIC cxx_std_11)
endif ()

if ("${CUDA_ENABLED}")
  set_target_properties (Coordinate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

file(GLOB_RECURSE COORDINATE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Coordinate ${COORDINATE_SOURCES})

if ("${CUDA_ENABLED}")
  set_target_properties (Coordinate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

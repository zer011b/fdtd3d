message ("Coordinate build.")

file(GLOB_RECURSE COORDINATE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if ("${CUDA_ENABLED}")
  cuda_add_library (Coordinate ${COORDINATE_SOURCES})
else ()
  add_library (Coordinate ${COORDINATE_SOURCES})
endif ()

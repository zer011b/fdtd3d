message ("Layout build.")

file(GLOB_RECURSE LAYOUT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if ("${CUDA_ENABLED}")
  cuda_add_library (Layout ${LAYOUT_SOURCES})
else ()
  add_library (Layout ${LAYOUT_SOURCES})
endif ()

target_link_libraries (Layout Coordinate)

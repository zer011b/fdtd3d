message ("Grid build.")

file(GLOB_RECURSE FIELD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Grid ${FIELD_SOURCES})

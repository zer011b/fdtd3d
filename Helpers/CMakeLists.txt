message ("Helpers build.")

file(GLOB_RECURSE HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Helpers ${HELPER_SOURCES})

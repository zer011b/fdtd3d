message ("Scheme build.")

file(GLOB_RECURSE SCHEME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Scheme ${SCHEME_SOURCES})

target_link_libraries (Scheme Layout Settings)

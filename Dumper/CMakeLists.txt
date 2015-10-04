message ("Dumper build.")

file(GLOB_RECURSE FIELD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (Dumper ${FIELD_SOURCES})
target_link_libraries (Dumper EasyBMP)

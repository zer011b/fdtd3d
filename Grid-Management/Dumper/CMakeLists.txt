message ("Dumper build.")

file(GLOB_RECURSE DUMPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (Dumper ${DUMPER_SOURCES})
target_link_libraries (Dumper EasyBMP)

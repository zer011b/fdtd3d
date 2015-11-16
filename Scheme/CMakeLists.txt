message ("Scheme build.")

file(GLOB_RECURSE LOADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (Scheme ${LOADER_SOURCES})

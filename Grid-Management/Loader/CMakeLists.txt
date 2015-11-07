message ("Loader build.")

file(GLOB_RECURSE LOADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (Loader ${LOADER_SOURCES})
target_link_libraries (Loader EasyBMP Helpers)

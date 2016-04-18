file(GLOB_RECURSE FILE_MANAGER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (FM ${FILE_MANAGER_SOURCES})
target_link_libraries (FM EasyBMP Helpers)

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/Loader")
add_subdirectory (Loader)

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/Dumper")
add_subdirectory (Dumper)

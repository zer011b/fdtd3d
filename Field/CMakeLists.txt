message ("Field build.")

file(GLOB_RECURSE FIELD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add EasyBMP library
add_library (Field ${FIELD_SOURCES})

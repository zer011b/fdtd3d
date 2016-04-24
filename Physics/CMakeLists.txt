message ("Physics build.")

file(GLOB_RECURSE PHYSICS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Physics ${PHYSICS_SOURCES})

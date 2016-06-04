message ("Kernels build.")

file(GLOB_RECURSE KERNELS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library (Kernels ${KERNELS_SOURCES})

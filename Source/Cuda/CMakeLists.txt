message ("Cuda kernels build.")

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_library (CudaScheme ${CUDA_SOURCES})
target_link_libraries (CudaScheme Layout Coordinate Kernels)

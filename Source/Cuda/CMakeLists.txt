message ("Cuda kernels build.")

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

cuda_add_library (CudaScheme ${CUDA_SOURCES})

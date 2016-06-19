message ("Cuda kernels build.")

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
set (CUDA_SOURCES ${CUDA_SOURCES} PARENT_SCOPE)

#add_library (CudaKernels ${CUDA_SOURCES})

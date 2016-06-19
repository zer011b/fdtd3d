#ifndef CUDA_GLOBAL_KERNELS_H
#define CUDA_GLOBAL_KERNELS_H

#include <cstdio>

#include "Kernels.h"

#define cudaCheckError() \
  { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) \
    { \
      fprintf(stderr, "Fatal error: %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  }

#define cudaCheckErrorCmd(cmd) \
  { \
    (cmd); \
    cudaCheckError(); \
  }

#define cudaCheckExitStatus(...) \
  { \
    __VA_ARGS__; \
    cudaCheckErrorCmd (cudaMemcpy (&exitStatus, exitStatusCuda, sizeof (CudaExitStatus), cudaMemcpyDeviceToHost)); \
    if (exitStatus != CUDA_OK) \
    { \
      cudaCheckError (); \
      *retval = CUDA_ERROR; \
      return; \
    } \
  }

__global__ void cudaCalculateTMzEStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue, FieldValue, grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzESource (CudaExitStatus *, FieldValue *, grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzHStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *, FieldValue, FieldValue, grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzHSource (CudaExitStatus *, FieldValue *, FieldValue *,
                                         grid_coord, grid_coord, time_step);

#endif /* !CUDA_GLOBAL_KERNELS_H */

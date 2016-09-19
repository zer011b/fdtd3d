#ifndef CUDA_GLOBAL_KERNELS_H
#define CUDA_GLOBAL_KERNELS_H

#include <cstdio>

#include "Kernels.h"
#include "CudaDefines.h"

__global__ void cudaCalculateTMzEStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzESource (CudaExitStatus *, FieldValue *, grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzHStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *, FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord, grid_coord, time_step);

__global__ void cudaCalculateTMzHSource (CudaExitStatus *, FieldValue *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord, grid_coord, time_step);

#endif /* !CUDA_GLOBAL_KERNELS_H */

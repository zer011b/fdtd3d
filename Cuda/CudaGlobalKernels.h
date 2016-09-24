#ifndef CUDA_GLOBAL_KERNELS_H
#define CUDA_GLOBAL_KERNELS_H

#include <cstdio>

#include "Kernels.h"
#include "CudaDefines.h"

__global__ void cudaCalculateTMzEzStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                        FieldValue *,
                                        FieldValue, FieldValue,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        time_step);

__global__ void cudaCalculateTMzEzSource (CudaExitStatus *, FieldValue *,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          time_step);

__global__ void cudaCalculateTMzHxStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                        FieldValue, FieldValue,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        time_step);

__global__ void cudaCalculateTMzHyStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                        FieldValue, FieldValue,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        time_step);

__global__ void cudaCalculateTMzHxSource (CudaExitStatus *, FieldValue *,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          time_step);

__global__ void cudaCalculateTMzHySource (CudaExitStatus *, FieldValue *,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          time_step);

#endif /* !CUDA_GLOBAL_KERNELS_H */

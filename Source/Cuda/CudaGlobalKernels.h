#ifndef CUDA_GLOBAL_KERNELS_H
#define CUDA_GLOBAL_KERNELS_H

#include <cstdio>

#include "Kernels.h"
#include "CudaDefines.h"

#include "GridCoordinate3D.h"

__global__ void cudaCalculateTMzEzStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                        FieldValue *,
                                        FieldValue, FieldValue,
                                        GridCoordinate3D,
                                        grid_coord, grid_coord,
                                        grid_coord, grid_coord,
                                        time_step);

__global__ void cudaCalculateTMzEzSource (CudaExitStatus *, FieldValue *,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          grid_coord, grid_coord,
                                          time_step,
                                          int processId);

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

__global__ void cudaCalculate3DExStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DEyStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DEzStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DHxStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DHyStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DHzStep (CudaExitStatus *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                                       FieldValue *,
                                       FieldValue, FieldValue,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       grid_coord, grid_coord, grid_coord,
                                       time_step);

__global__ void cudaCalculate3DExSource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);

__global__ void cudaCalculate3DEySource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);

__global__ void cudaCalculate3DEzSource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);

__global__ void cudaCalculate3DHxSource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);

__global__ void cudaCalculate3DHySource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);

__global__ void cudaCalculate3DHzSource (CudaExitStatus *, FieldValue *,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         grid_coord, grid_coord, grid_coord,
                                         time_step,
                                         int processId);
#endif /* !CUDA_GLOBAL_KERNELS_H */

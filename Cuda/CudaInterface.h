#ifndef CUDA_KERNEL_INTERFACE_H
#define CUDA_KERNEL_INTERFACE_H

#include "Kernels.h"
#include "CudaGlobalKernels.h"

void cudaExecute2DTMzSteps (CudaExitStatus *,
                            FieldValue *, FieldValue *, FieldValue *,
                            FieldValue *, FieldValue *, FieldValue *,
                            FieldValue *, FieldValue *,
                            FieldValue, FieldValue, grid_coord, grid_coord,
                            time_step, time_step, uint32_t, uint32_t, uint32_t, uint32_t);

#endif /* !CUDA_KERNEL_INTERFACE_H */

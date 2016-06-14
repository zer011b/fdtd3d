#ifndef EXECUTE_H
#define EXECUTE_H

#include "Kernels.h"

void cudaExecuteTMzSteps (CudaExitStatus *,
                          FieldValue *, FieldValue *, FieldValue *,
                          FieldValue *, FieldValue *, FieldValue *,
                          FieldValue, FieldValue, grid_coord, grid_coord,
                          time_step, time_step, uint32_t, uint32_t, uint32_t, uint32_t);

#endif

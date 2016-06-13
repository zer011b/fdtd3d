#ifndef EXECUTE_H
#define EXECUTE_H

#include "Kernels.h"

/*extern void executeTMz (FieldValue *, FieldValue *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                        grid_coord, grid_coord, FieldValue, FieldValue, int);

extern void executeTMzStep (FieldValue *, FieldValue *, FieldValue *, FieldValue *, FieldValue *, FieldValue *,
                            grid_coord, grid_coord, FieldValue, FieldValue, int);*/

CudaExitStatus cudaExecuteTMzSteps (FieldValue *, FieldValue *, FieldValue *,
                                    FieldValue *, FieldValue *, FieldValue *,
                                    FieldValue, FieldValue, grid_coord, grid_coord,
                                    time_step, time_step, uint32_t, uint32_t, uint32_t, uint32_t);

#endif

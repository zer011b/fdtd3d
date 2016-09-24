#ifndef CUDA_KERNEL_INTERFACE_H
#define CUDA_KERNEL_INTERFACE_H

#include "Kernels.h"
#include "CudaDefines.h"
#include "GridLayout.h"

#ifdef PARALLEL_GRID
#include "ParallelGrid.h"
#else
#include "Grid.h"
#endif

#if defined (PARALLEL_GRID)
void cudaExecute2DTMzSteps (CudaExitStatus *,
                            YeeGridLayout &,
                            FieldValue, FieldValue,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            time_step, time_step,
                            int);
#else
void cudaExecute2DTMzSteps (CudaExitStatus *,
                            YeeGridLayout &,
                            FieldValue, FieldValue,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            time_step, time_step,
                            int);
#endif

#endif /* !CUDA_KERNEL_INTERFACE_H */

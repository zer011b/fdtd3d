#ifndef CUDA_KERNEL_INTERFACE_H
#define CUDA_KERNEL_INTERFACE_H

#include "Kernels.h"
#include "CudaDefines.h"
#include "YeeGridLayout.h"

#ifdef PARALLEL_GRID
#include "ParallelGrid.h"
#else
#include "Grid.h"
#endif

#ifdef PARALLEL_GRID
void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &,
                            FieldValue, FieldValue,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            ParallelGrid &,
                            time_step,
                            int);

void cudaExecute3DSteps (CudaExitStatus *retval,
                         YeeGridLayout &,
                         FieldValue, FieldValue,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         ParallelGrid &,
                         time_step,
                         int);
#else
void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &,
                            FieldValue, FieldValue,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            Grid<GridCoordinate2D> &,
                            time_step,
                            int);

void cudaExecute3DSteps (CudaExitStatus *retval,
                         YeeGridLayout &,
                         FieldValue, FieldValue,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         Grid<GridCoordinate3D> &,
                         time_step,
                         int);
#endif

void cudaInit (int);
void cudaInfo ();

#endif /* !CUDA_KERNEL_INTERFACE_H */

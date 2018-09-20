#ifndef GRID_INTERFACE_H
#define GRID_INTERFACE_H

#include "Grid.h"

#ifdef CUDA_SOURCES
#include "cudaGrid.h"
#else /* CUDA_SOURCES */
#include "ParallelGrid.h"
#endif /* !CUDA_SOURCES */


#endif /* GRID_INTERFACE_H */

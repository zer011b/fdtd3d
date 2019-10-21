#ifndef GRID_INTERFACE_H
#define GRID_INTERFACE_H

#include "Grid.h"
#include "ParallelGrid.h"

#ifdef CUDA_SOURCES
#include "CudaGrid.h"
#endif /* CUDA_SOURCES */

/*
 * CPU/GPU grids interface for next time step.
 *
 * ==== I. CPU grid next time step ====
 *
 *   parallel/sequential:
 *   1) N x grid->shiftInTime
 *   parallel:
 *   2) 1 x group->nextShareStep
 *   3) N x grid->share
 *
 * ==== II. GPU grid next time step ====
 *
 *   1) N x grid->shiftInTime <<< >>> kernel   -> shifts gpu arrays
 *   2) N x grid->shiftInTime                  -> shifts cpu arrays
 *   3) N x grid->nextShareStep                -> increments cpu counters
 *   4) share is done with CPU after N time steps
 */

#endif /* GRID_INTERFACE_H */

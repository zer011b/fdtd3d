/*
 * Copyright (C) 2015 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

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

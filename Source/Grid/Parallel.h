/*
 * Copyright (C) 2018 Gleb Balykov
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

#ifndef PARALLEL_H
#define PARALLEL_H

#ifdef PARALLEL_GRID

#include <mpi.h>

/**
 * Base grid of parallel grid and parallel grid coordinate
 */
#ifdef GRID_1D
#define ParallelGridBase Grid<GridCoordinate1D>
#define ParallelGridCoordinateTemplate GridCoordinate1DTemplate
#define ParallelGridCoordinate GridCoordinate1D
#define ParallelGridCoordinateFP GridCoordinateFP1D
#endif /* GRID_1D */

#ifdef GRID_2D
#define ParallelGridBase Grid<GridCoordinate2D>
#define ParallelGridCoordinateTemplate GridCoordinate2DTemplate
#define ParallelGridCoordinate GridCoordinate2D
#define ParallelGridCoordinateFP GridCoordinateFP2D
#endif /* GRID_2D */

#ifdef GRID_3D
#define ParallelGridBase Grid<GridCoordinate3D>
#define ParallelGridCoordinateTemplate GridCoordinate3DTemplate
#define ParallelGridCoordinate GridCoordinate3D
#define ParallelGridCoordinateFP GridCoordinateFP3D
#endif /* GRID_3D */

/**
 * Process ID for non-existing processes
 */
#define PID_NONE (-1)

/**
 * Parallel grid buffer types.
 */
enum BufferPosition
{
#define FUNCTION(X) X,
#include "BufferPosition.inc.h"
}; /* BufferPosition */

#endif /* PARALLEL_GRID */

#endif /* !PARALLEL_H */

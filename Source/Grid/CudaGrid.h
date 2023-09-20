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

#ifndef CUDA_GRID_H
#define CUDA_GRID_H

#include "PAssert.h"
#include "FieldValue.h"
#include "GridCoordinate3D.h"
#include "Settings.h"
#include "Grid.h"

#ifdef CUDA_ENABLED

/**
 * Cuda version of CPU Grid<TCoord>
 *
 * Similar to ParallelGrid in that both of them store only part of whole grid (and both have buffers).
 * Difference is that ParallelGrid's sizes and buffers are determined based on virtual topology.
 * CudaGrid size is determined as the size of block, which GPU can fit. CudaGrid buffer always exists for all axes,
 * in contrast to ParallelGrid, which has buffers only for axis, which is spread between processes.
 *
 * Grid 6x6       ->  CudaGrid for blockSize (3x3) and bufSize (b,b), S cell is the startOfBlock
 * -------------  ->  -----------
 * | | | | | | |  ->  |b|b|b|b|b|
 * -------------  ->  -----------
 * | | | | | | |  ->  |b|S| | |b|
 * -------------  ->  -----------
 * | | | | | | |  ->  |b| | | |b|
 * -------------  ->  -----------
 * | | | | | | |  ->  |b| | | |b|
 * -------------  ->  -----------
 * | | | | | | |  ->  |b|b|b|b|b|
 * -------------  ->  -----------
 * | | | | | | |  ->
 * -------------  ->
 *
 * For the first block S will be (0,0) and left buffers should be never accessed.
 * For the second block S will be (3,0), for the third (0,3), and (3,3) for the last block.
 *
 * For the block of size (6,6) situation is the same, CudaGrid will have size (6+b,6+b) and buffers should be never used
 *
 * -------- Coordinate systems --------
 * 1. Total coordinate system is the global coordinate system, considering all computational nodes.
 * 2. Relative coordinate system starts from the start of the GPU chunk (considering buffers!),
 *    stored in this grid (i.e. at startOfBlock - bufSize). When CudaGrid stores full grid from CPU in sequential mode,
 *    buffers are empty and total coordinate of left buffer will be -1, but this is ok,
 *    since buffers should never be used.
 * 3. Coordinate system of blocks, which is relative to CPU chunk, not considering buffers,
 *    i.e. starts at startOfBlock. This is useful, because in this coordinate system both for sequential and
 *    parallel modes first block has (0,0) coordinates.
 *    NOTE: values in this coordinate system are only passed to copyToCPU and copyFromCPU!
 */
template <class TCoord>
class CudaGrid
{
protected:

  /**
   * Size block of CPU grid, which can be stored in this Cuda grid
   */
  TCoord sizeOfBlock;

  /**
   * Absolute coordinate of start of this block in CPU grid (not relative!) (this doesn't consider buffers)
   */
  TCoord startOfBlock;

  /**
   * Absolute coordinate of end of this block in CPU grid (not relative!) (this doesn't consider buffers)
   */
  TCoord endOfBlock;

  /**
   * Size of buffer
   */
  TCoord bufSize;

  /**
   * Size of grid, which is actually allocated, i.e. sum of sizeOfBlock and bufSize
   */
  TCoord size;

  /**
   * Capacity of gridValues array (this should match size)
   */
  grid_coord sizeGridValues;

  /**
   * Number of stored time steps
   */
  int storedSteps;

  /**
   * Total size of CPU grid
   */
  TCoord totalSize;

  /**
   * Corresponding CPU grid
   */
  Grid<TCoord> *cpuGrid;

  /**
   * Vector of points in grid, containing GPU pointers to arrays
   *
   * d_gridValue -> FieldValue*, FieldValue*, ... , FieldValue*  | GPU
   *                     |            |                  |       |----
   *                   arrays       arrays             arrays    | GPU
   */
  FieldValue **d_gridValues;

  /**
   * Vector of points in grid, containing CPU pointers to arrays
   *
   * gridValuesDevicePointers -> FieldValue*, FieldValue*, ... , FieldValue*  | CPU
   *                                  |            |                  |       |----
   *                                arrays       arrays             arrays    | GPU
   */
  FieldValue **gridValuesDevicePointers;

  /**
   * Helper array, which is used for copying data from/to GPU
   */
  FieldValue *helperGridValues;

  /**
   * Step at which to perform share operations for synchronization of computational nodes
   */
  time_step shareStep;

  TCoord hasLeft;
  TCoord hasRight;

  /*
   * TODO: add debug uninitialized flag
   */

protected:

  static CUDA_DEVICE CUDA_HOST bool isLegitIndex (const TCoord &, const TCoord &);
  static CUDA_DEVICE CUDA_HOST grid_coord calculateIndexFromPosition (const TCoord &, const TCoord &);

private:

  CUDA_HOST CudaGrid (const CudaGrid &);
  CUDA_HOST CudaGrid<TCoord> & operator = (const CudaGrid<TCoord> &);

  CUDA_HOST bool checkParams ();
  CUDA_DEVICE CUDA_HOST void shift (FieldValue **);

protected:

  CUDA_DEVICE CUDA_HOST bool isLegitIndex (const TCoord &) const;

public:

  CUDA_HOST CudaGrid (const TCoord &, const TCoord &, Grid<TCoord> *);
  CUDA_HOST ~CudaGrid ();

  CUDA_HOST void copyFromCPU (const TCoord &, const TCoord &);
  CUDA_HOST void copyToCPU ();

  CUDA_DEVICE CUDA_HOST const TCoord & getSize () const;
  CUDA_DEVICE CUDA_HOST const TCoord & getBufSize () const;
  CUDA_DEVICE CUDA_HOST grid_coord getSizeGridValues () const;

  CUDA_DEVICE CUDA_HOST time_step getShareStep () const;

  CUDA_DEVICE CUDA_HOST TCoord getTotalSize () const;
  CUDA_DEVICE CUDA_HOST
  TCoord getTotalPosition (const TCoord & pos) const;
  CUDA_DEVICE CUDA_HOST
  TCoord getRelativePosition (const TCoord & pos) const;

  CUDA_DEVICE CUDA_HOST
  bool hasValueForCoordinate (const TCoord &position) const;

  CUDA_DEVICE CUDA_HOST
  FieldValue * getFieldValueByAbsolutePos (const TCoord &absPosition,
                                           int time_step_back)
  {
    return getFieldValue (getRelativePosition (absPosition), time_step_back);
  }

  CUDA_DEVICE CUDA_HOST
  FieldValue * getFieldValueOrNullByAbsolutePos (const TCoord &absPosition,
                                                 int time_step_back)
  {
    if (!hasValueForCoordinate (absPosition))
    {
      return NULLPTR;
    }

    return getFieldValueByAbsolutePos (absPosition, time_step_back);
  } /* getFieldValueOrNullByAbsolutePos */

  CUDA_DEVICE CUDA_HOST TCoord getComputationStart (const TCoord &) const;
  CUDA_DEVICE CUDA_HOST TCoord getComputationEnd (const TCoord &) const;
  CUDA_DEVICE CUDA_HOST TCoord calculatePositionFromIndex (grid_coord) const;
  CUDA_DEVICE CUDA_HOST grid_coord calculateIndexFromPosition (const TCoord &) const;

  CUDA_DEVICE void setFieldValue (const FieldValue &, const TCoord &, int);
  CUDA_DEVICE void setFieldValue (const FieldValue &, grid_coord, int);
  CUDA_DEVICE CUDA_HOST FieldValue * getFieldValue (const TCoord &, int);
  CUDA_DEVICE CUDA_HOST FieldValue * getFieldValue (grid_coord, int);

  CUDA_DEVICE CUDA_HOST void shiftInTime ();

  CUDA_HOST void nextShareStep ();
  CUDA_HOST void zeroShareStep ();

  CUDA_DEVICE CUDA_HOST
  TCoord getHasLeft () const
  {
    return hasLeft;
  }
  CUDA_DEVICE CUDA_HOST
  TCoord getHasRight () const
  {
    return hasRight;
  }

  CUDA_DEVICE CUDA_HOST
  bool isBufferLeftPosition (const TCoord & pos)
  {
    if (pos >= startOfBlock)
    {
      return false;
    }

    return true;
  }

  CUDA_DEVICE CUDA_HOST
  bool isBufferRightPosition (const TCoord & pos)
  {
    if (pos < endOfBlock)
    {
      return false;
    }

    return true;
  }
}; /* CudaGrid */

/**
 * Constructor of Cuda grid with size of block
 */
template <class TCoord>
CUDA_HOST
CudaGrid<TCoord>::CudaGrid (const TCoord & s, /**< size of this Cuda grid */
                            const TCoord & buf, /**< size of buffer */
                            Grid<TCoord> *grid) /**< corresponding CPU grid */
  : sizeOfBlock (s)
  , startOfBlock (TCoord ())
  , endOfBlock (TCoord ())
  , bufSize (buf)
  , size (sizeOfBlock + bufSize * 2)
  , sizeGridValues (size.calculateTotalCoord ())
  , storedSteps (grid->getCountStoredSteps ())
  , totalSize (grid->getTotalSize ())
  , cpuGrid (grid)
  , d_gridValues (NULLPTR)
  , gridValuesDevicePointers (NULLPTR)
  , helperGridValues (NULLPTR)
  , shareStep (0)
  , hasLeft (TCoord ())
  , hasRight (TCoord ())
{
  ASSERT (checkParams ());
  ASSERT (storedSteps > 0);
  ASSERT (sizeGridValues > 0);

  gridValuesDevicePointers = new FieldValue * [storedSteps];

  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gridValues, storedSteps * sizeof (FieldValue *)));
  for (int i = 0; i < storedSteps; ++i)
  {
    FieldValue *d_tmp = NULLPTR;
    cudaCheckErrorCmd (cudaMalloc ((void **) &d_tmp, sizeGridValues * sizeof (FieldValue)));
    cudaCheckErrorCmd (cudaMemcpy (d_gridValues + i, &d_tmp, sizeof (FieldValue *), cudaMemcpyHostToDevice));

    gridValuesDevicePointers[i] = d_tmp;
  }

  helperGridValues = new FieldValue [sizeGridValues];

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New Cuda grid '%s' with %d stored steps and with raw size: " C_MOD ".\n",
    grid->getName (), storedSteps, sizeGridValues);
} /* CudaGrid<TCoord>::CudaGrid */

/**
 * Destructor of Cuda grid. Should delete all field point values
 */
template <class TCoord>
CUDA_HOST
CudaGrid<TCoord>::~CudaGrid ()
{
  for (int i = 0; i < storedSteps; ++i)
  {
    cudaCheckErrorCmd (cudaFree (gridValuesDevicePointers[i]));
  }
  cudaCheckErrorCmd (cudaFree (d_gridValues));

  delete[] gridValuesDevicePointers;
  delete[] helperGridValues;
} /* CudaGrid<TCoord>::~CudaGrid */

/**
 * Replace previous time layer with current and so on
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
void
CudaGrid<TCoord>::shiftInTime ()
{
#ifdef __CUDA_ARCH__
  shift (d_gridValues);
#else
  shift (gridValuesDevicePointers);

  nextShareStep ();
#endif
} /* CudaGrid<TCoord>::shiftInTime */

template <class TCoord>
CUDA_DEVICE CUDA_HOST
void
CudaGrid<TCoord>::shift (FieldValue **grid)
{
  /*
   * Reuse oldest grid as new current
   */
  ASSERT (storedSteps > 0);

  FieldValue *oldest = grid[storedSteps - 1];

  for (int i = storedSteps - 1; i >= 1; --i)
  {
    grid[i] = grid[i - 1];
  }

  grid[0] = oldest;
}

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
bool
CudaGrid<TCoord>::isLegitIndex (const TCoord& position) const /**< coordinate in grid */
{
  return isLegitIndex (position, size);
} /* CudaGrid<TCoord>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from N-dimensional position
 *
 * @return one-dimensional coordinate from N-dimensional position
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
grid_coord
CudaGrid<TCoord>::calculateIndexFromPosition (const TCoord& position) const /**< coordinate in grid */
{
  return calculateIndexFromPosition (position, size);
} /* CudaGrid<TCoord>::calculateIndexFromPosition */

/**
 * Get size of the grid
 *
 * @return size of the grid
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
const TCoord &
CudaGrid<TCoord>::getSize () const
{
  return size;
} /* CudaGrid<TCoord>::getSize */

/**
 * Get size of buffer
 *
 * @return size of buffer
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
const TCoord &
CudaGrid<TCoord>::getBufSize () const
{
  return bufSize;
} /* CudaGrid<TCoord>::getBufSize */

/**
 * Get capacity of the grid
 *
 * @return capacity of the grid
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
grid_coord
CudaGrid<TCoord>::getSizeGridValues () const
{
  return sizeGridValues;
} /* CudaGrid<TCoord>::getSizeGridValues */

/**
 * Get share step
 *
 * @return share step
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
time_step
CudaGrid<TCoord>::getShareStep () const
{
  return shareStep;
} /* CudaGrid<TCoord>::getSizeGridValues */

/**
 * Set field value at coordinate in grid
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::setFieldValue (const FieldValue & value, /**< field value */
                                 const TCoord &position, /**< coordinate in grid */
                                 int time_step_back) /**< shift in time */
{
  ASSERT (isLegitIndex (position));
  ASSERT (time_step_back < storedSteps);

  grid_coord coord = calculateIndexFromPosition (position);

  d_gridValues[time_step_back][coord] = value;
} /* CudaGrid<TCoord>::setFieldValue */

/**
 * Set field value at coordinate in grid
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::setFieldValue (const FieldValue & value, /**< field value */
                                 grid_coord coord, /**< coordinate in grid */
                                 int time_step_back) /**< shift in time */
{
  ASSERT (coord >= 0 && coord < sizeGridValues);
  ASSERT (time_step_back < storedSteps);

  d_gridValues[time_step_back][coord] = value;
} /* CudaGrid<TCoord>::setFieldValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
FieldValue *
CudaGrid<TCoord>::getFieldValue (const TCoord &position, /**< coordinate in grid */
                                 int time_step_back) /**< shift in time */
{
  ASSERT (isLegitIndex (position));

  grid_coord coord = calculateIndexFromPosition (position);

  return getFieldValue (coord, time_step_back);
} /* CudaGrid<TCoord>::getFieldValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
FieldValue *
CudaGrid<TCoord>::getFieldValue (grid_coord coord, /**< index in grid */
                                 int time_step_back) /**< shift in time */
{
  ASSERT (coord >= 0 && coord < getSizeGridValues ());

#ifdef __CUDA_ARCH__
  return &d_gridValues[time_step_back][coord];
#else /* __CUDA_ARCH__ */
  return gridValuesDevicePointers[time_step_back] + coord;
#endif /* !__CUDA_ARCH__ */
} /* CudaGrid<TCoord>::getFieldValue */

/**
 * Increase share step
 */
template <class TCoord>
CUDA_HOST
void
CudaGrid<TCoord>::nextShareStep ()
{
  ++shareStep;
} /* CudaGrid<TCoord>::nextShareStep */

/**
 * Set share step to zero
 */
template <class TCoord>
CUDA_HOST
void
CudaGrid<TCoord>::zeroShareStep ()
{
  shareStep = 0;
} /* CudaGrid<TCoord>::zeroShareStep */

/**
 * Get total size of grid. Is equal to size in non-parallel grid
 *
 * @return total size of grid
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
TCoord
CudaGrid<TCoord>::getTotalSize () const
{
  return totalSize;
} /* CudaGrid<TCoord>::getTotalSize */

#endif /* CUDA_ENABLED */

#endif /* !CUDA_GRID_H */

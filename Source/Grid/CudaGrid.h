#ifndef CUDA_GRID_H
#define CUDA_GRID_H

#include "Assert.h"
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
   * Vector of points in grid
   */
  FieldValue **d_gridValues;
  FieldValue **gridValuesDevicePointers;

  FieldValue *helperGridValues;

#ifdef DEBUG_INFO
  /**
   * Current time step.
   */
  time_step timeStep;
#endif /* DEBUG_INFO */

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

  CUDA_DEVICE void shiftInTime ();
  CUDA_HOST void nextTimeStep ();
  CUDA_HOST time_step getTimeStep ();

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
#ifdef DEBUG_INFO
  , timeStep (0)
#endif /* DEBUG_INFO */
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New Cuda grid '%s' with %d stored steps and with raw size: %llu.\n",
    grid->getName (), storedSteps, (unsigned long long)sizeGridValues);
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

  delete[] helperGridValues;
} /* CudaGrid<TCoord>::~CudaGrid */

/**
 * Replace previous time layer with current and so on
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::shiftInTime ()
{
  /*
   * Reuse oldest grid as new current
   */
  shift (d_gridValues);
} /* CudaGrid<TCoord>::shiftInTime */

template <class TCoord>
CUDA_DEVICE CUDA_HOST
void
CudaGrid<TCoord>::shift (FieldValue **grid)
{
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
 * Switch to next time step
 */
template <class TCoord>
CUDA_HOST
void
CudaGrid<TCoord>::nextTimeStep ()
{
  shift (gridValuesDevicePointers);

  nextShareStep ();

#ifdef DEBUG_INFO
  ++timeStep;
#endif /* DEBUG_INFO */
} /* CudaGrid<TCoord>::nextTimeStep */

#ifdef DEBUG_INFO

/**
 * Set time step
 */
template <class TCoord>
CUDA_HOST
time_step
CudaGrid<TCoord>::getTimeStep ()
{
  return timeStep;
} /* CudaGrid<TCoord>::getTimeStep */

#endif /* DEBUG_INFO */

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

#ifndef CUDA_GRID_H
#define CUDA_GRID_H

#include "Assert.h"
#include "FieldPoint.h"
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
   * Coordinate of start of this block in CPU grid
   */
  TCoord startOfBlock;

  /**
   * Coordinate of end of this block in CPU grid
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
   * Corresponding CPU grid
   */
  Grid<TCoord> *cpuGrid;

  /**
   * Vector of points in grid. Owns this. Deletes all FieldPointValue* itself.
   */
  FieldPointValue **d_gridValues;

  /**
   * Current time step.
   */
  time_step timeStep;

  /**
   * Step at which to perform share operations for synchronization of computational nodes
   */
  time_step shareStep;

  /*
   * TODO: add debug uninitialized flag
   */

protected:

  static CUDA_DEVICE CUDA_HOST bool isLegitIndex (const TCoord &, const TCoord &);
  static CUDA_DEVICE CUDA_HOST grid_coord calculateIndexFromPosition (const TCoord &, const TCoord &);

private:

  CUDA_DEVICE void shiftInTime ();

  CUDA_HOST CudaGrid (const CudaGrid &);
  CUDA_HOST CudaGrid<TCoord> & operator = (const CudaGrid<TCoord> &);

  CUDA_HOST bool checkParams ();

protected:

  CUDA_DEVICE CUDA_HOST bool isLegitIndex (const TCoord &) const;
  CUDA_DEVICE CUDA_HOST grid_coord calculateIndexFromPosition (const TCoord &) const;

public:

  CUDA_HOST CudaGrid (const TCoord &, const TCoord &, Grid<TCoord> *);
  CUDA_HOST ~CudaGrid ();

  CUDA_HOST void copyFromCPU (const TCoord &, const TCoord &);
  CUDA_HOST void copyToCPU ();

  CUDA_DEVICE CUDA_HOST const TCoord & getSize () const;
  CUDA_DEVICE CUDA_HOST const TCoord & getBufSize () const;
  CUDA_DEVICE CUDA_HOST grid_coord getSizeGridValues () const;
  CUDA_DEVICE CUDA_HOST time_step getShareStep () const;

  CUDA_DEVICE TCoord getComputationStart (TCoord) const;
  CUDA_DEVICE TCoord getComputationEnd (TCoord) const;
  CUDA_DEVICE CUDA_HOST TCoord calculatePositionFromIndex (grid_coord) const;

  CUDA_DEVICE void setFieldPointValue (FieldPointValue *, const TCoord &);
  CUDA_DEVICE FieldPointValue * getFieldPointValue (const TCoord &);
  CUDA_DEVICE FieldPointValue * getFieldPointValue (grid_coord);

  CUDA_DEVICE void nextTimeStep ();
  CUDA_DEVICE CUDA_HOST void setTimeStep (time_step);

  CUDA_DEVICE void nextShareStep ();
  CUDA_DEVICE CUDA_HOST void zeroShareStep ();
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
  , cpuGrid (grid)
  , d_gridValues (NULLPTR)
  , timeStep (0)
  , shareStep (0)
{
  ASSERT (checkParams ());

  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gridValues, sizeGridValues * sizeof (FieldPointValue *)));

  for (grid_coord i = 0; i < sizeGridValues; ++i)
  {
    FieldPointValue *d_val;
    cudaCheckErrorCmd (cudaMalloc ((void **) &(d_val), sizeof (FieldPointValue)));
    cudaCheckErrorCmd (cudaMemcpy (&d_gridValues[i], &d_val, sizeof (FieldPointValue *), cudaMemcpyHostToDevice));
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New Cuda grid '%s' with raw size: %lu.\n", grid->getName (), sizeGridValues);
} /* CudaGrid<TCoord>::CudaGrid */

/**
 * Destructor of Cuda grid. Should delete all field point values
 */
template <class TCoord>
CUDA_HOST
CudaGrid<TCoord>::~CudaGrid ()
{
  for (grid_coord i = 0; i < sizeGridValues; ++i)
  {
    FieldPointValue *d_val = NULLPTR;
    cudaCheckErrorCmd (cudaMemcpy (&d_val, &d_gridValues[i], sizeof (FieldPointValue *), cudaMemcpyDeviceToHost));

    ASSERT (d_val != NULLPTR);
    cudaCheckErrorCmd (cudaFree (d_val));
  }
  cudaCheckErrorCmd (cudaFree (d_gridValues));
} /* CudaGrid<TCoord>::~CudaGrid */

/**
 * Replace previous time layer with current and so on
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::shiftInTime ()
{
  for (grid_coord iter = 0; iter < getSizeGridValues (); ++iter)
  {
    ASSERT (d_gridValues[iter] != NULLPTR);
    d_gridValues[iter]->shiftInTime ();
  }
} /* CudaGrid<TCoord>::shiftInTime */

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
 * Set field point value at coordinate in grid
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::setFieldPointValue (FieldPointValue *value, /**< field point value */
                                      const TCoord &position) /**< coordinate in grid */
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_coord coord = calculateIndexFromPosition (position);

  ASSERT (d_gridValues[coord] == NULLPTR);
  d_gridValues[coord] = value;
} /* CudaGrid<TCoord>::setFieldPointValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
CUDA_DEVICE
FieldPointValue *
CudaGrid<TCoord>::getFieldPointValue (const TCoord &position) /**< coordinate in grid */
{
  ASSERT (isLegitIndex (position));

  grid_coord coord = calculateIndexFromPosition (position);

  return getFieldPointValue (coord);
} /* CudaGrid<TCoord>::getFieldPointValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
CUDA_DEVICE
FieldPointValue *
CudaGrid<TCoord>::getFieldPointValue (grid_coord coord) /**< index in grid */
{
  ASSERT (coord >= 0 && coord < getSizeGridValues ());

  FieldPointValue* value = d_gridValues[coord];

  ASSERT (value);

  return value;
} /* CudaGrid<TCoord>::getFieldPointValue */

/**
 * Switch to next time step
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::nextTimeStep ()
{
  shiftInTime ();
  nextShareStep ();

  ASSERT (getShareStep () <= getBufSize ().get1 ());
  bool is_share_time = getShareStep () == getBufSize ().get1 ();

  if (is_share_time)
  {
    /*
     * Time to copy back to CPU. If copy hasn't happened, assert above will fire due to increase of shareStep.
     */
  }
} /* CudaGrid<TCoord>::nextTimeStep */

/**
 * Set time step
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
void
CudaGrid<TCoord>::setTimeStep (time_step step)
{
  timeStep = step;
} /* CudaGrid<TCoord>::setTimeStep */

/**
 * Increase share step
 */
template <class TCoord>
CUDA_DEVICE
void
CudaGrid<TCoord>::nextShareStep ()
{
  ++shareStep;
} /* CudaGrid<TCoord>::nextShareStep */

/**
 * Set share step to zero
 */
template <class TCoord>
CUDA_DEVICE CUDA_HOST
void
CudaGrid<TCoord>::zeroShareStep ()
{
  shareStep = 0;
} /* CudaGrid<TCoord>::zeroShareStep */

#endif /* CUDA_ENABLED */

#endif /* !CUDA_GRID_H */

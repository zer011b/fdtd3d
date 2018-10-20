#define CUDA_SOURCES

#include "CudaGrid.h"

#ifdef CUDA_ENABLED

/**
 * Check params for consistency
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate1D>::checkParams ()
{
  return true;
} /* CudaGrid<GridCoordinate1D>::checkParams */

/**
 * Check params for consistency
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate2D>::checkParams ()
{
  return bufSize.get1 () == bufSize.get2 ();
} /* CudaGrid<GridCoordinate2D>::checkParams */

/**
 * Check params for consistency
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate3D>::checkParams ()
{
  return bufSize.get1 () == bufSize.get2 ()
         && bufSize.get1 () == bufSize.get3 ();
} /* CudaGrid<GridCoordinate3D>::checkParams */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getComputationStart (GridCoordinate1D diffPosStart) const /**< offset from the left border */
{
  return GridCoordinate1D (getShareStep () + 1 + diffPosStart.get1 ()
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate1D>::getComputationStart */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getComputationStart (GridCoordinate2D diffPosStart) const /**< offset from the left border */
{
  return GridCoordinate2D (getShareStep () + 1 + diffPosStart.get1 (),
                           getShareStep () + 1 + diffPosStart.get2 ()
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
                           , getSize ().getType2 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate2D>::getComputationStart */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getComputationStart (GridCoordinate3D diffPosStart) const /**< offset from the left border */
{
  return GridCoordinate3D (getShareStep () + 1 + diffPosStart.get1 (),
                           getShareStep () + 1 + diffPosStart.get2 (),
                           getShareStep () + 1 + diffPosStart.get3 ()
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
                           , getSize ().getType2 ()
                           , getSize ().getType3 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate3D>::getComputationStart */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getComputationEnd (GridCoordinate1D diffPosEnd) const
{
  return getSize () - GridCoordinate1D (getShareStep () + 1 + diffPosEnd.get1 ()
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate1D>::getComputationEnd () */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getComputationEnd (GridCoordinate2D diffPosEnd) const
{
  return getSize () - GridCoordinate2D (getShareStep () + 1 + diffPosEnd.get1 (),
                                        getShareStep () + 1 + diffPosEnd.get2 ()
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate2D>::getComputationEnd () */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getComputationEnd (GridCoordinate3D diffPosEnd) const
{
  return getSize () - GridCoordinate3D (getShareStep () + 1 + diffPosEnd.get1 (),
                                        getShareStep () + 1 + diffPosEnd.get2 (),
                                        getShareStep () + 1 + diffPosEnd.get3 ()
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
                                        , getSize ().getType3 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate3D>::getComputationEnd () */

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate1D>::copyFromCPU (const GridCoordinate1D &start, /**< start coordinate of block to copy */
                                         const GridCoordinate1D &end) /**< end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;
  timeStep = cpuGrid->getTimeStep ();

  GridCoordinate1D zero (0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate1D startWithBuf = GridCoordinate1D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate1D endWithBuf = GridCoordinate1D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    GridCoordinate1D pos = GridCoordinate1D (index1
#ifdef DEBUG_INFO
                                             , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                                             );
    grid_coord index = calculateIndexFromPosition (GridCoordinate1D (c1
#ifdef DEBUG_INFO
                                                                     , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                                                                     ));

    ASSERT (index >= 0 && index < sizeGridValues);
    
    cudaCheckErrorCmd (cudaMemcpy (&d_gridValues[index], cpuGrid->getFieldPointValue (pos), sizeof (FieldPointValue), cudaMemcpyHostToDevice));
  }
} /* CudaGrid<GridCoordinate1D>::copyFromCPU */

/**
 * Copy to CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate1D>::copyToCPU ()
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate1D zero (0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate1D startWithBuf = GridCoordinate1D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate1D endWithBuf = GridCoordinate1D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    GridCoordinate1D pos = GridCoordinate1D (index1
#ifdef DEBUG_INFO
                                             , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                                             );
    grid_coord index = calculateIndexFromPosition (GridCoordinate1D (c1
#ifdef DEBUG_INFO
                                                   , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                                                   ));

    ASSERT (index >= 0 && index < sizeGridValues);

    cudaCheckErrorCmd (cudaMemcpy (cpuGrid->getFieldPointValue (pos), &d_gridValues[index], sizeof (FieldPointValue), cudaMemcpyDeviceToHost));
  }
} /* CudaGrid<GridCoordinate1D>::copyToCPU */

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate2D>::copyFromCPU (const GridCoordinate2D &start, /**< start coordinate of block to copy */
                                         const GridCoordinate2D &end) /**< end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;
  timeStep = cpuGrid->getTimeStep ();

  GridCoordinate2D zero (0, 0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 (), startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate2D startWithBuf = GridCoordinate2D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate2D endWithBuf = GridCoordinate2D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
    {
      grid_coord c2 = index2 - startWithBuf.get2 ();
      GridCoordinate2D pos = GridCoordinate2D (index1, index2
#ifdef DEBUG_INFO
                                               , startOfBlock.getType1 ()
                                               , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                                               );
      grid_coord index = calculateIndexFromPosition (GridCoordinate2D (c1, c2
#ifdef DEBUG_INFO
                                                     , startOfBlock.getType1 ()
                                                     , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                                                     ));

      ASSERT (index >= 0 && index < sizeGridValues);
      
      cudaCheckErrorCmd (cudaMemcpy (&d_gridValues[index], cpuGrid->getFieldPointValue (pos), sizeof (FieldPointValue), cudaMemcpyHostToDevice));
    }
  }
} /* CudaGrid<GridCoordinate2D>::copyFromCPU */

/**
 * Copy to CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate2D>::copyToCPU ()
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate2D zero (0, 0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 (), startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate2D startWithBuf = GridCoordinate2D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate2D endWithBuf = GridCoordinate2D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
    {
      grid_coord c2 = index2 - startWithBuf.get2 ();
      GridCoordinate2D pos = GridCoordinate2D (index1, index2
#ifdef DEBUG_INFO
                                               , startOfBlock.getType1 ()
                                               , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                                               );
      grid_coord index = calculateIndexFromPosition (GridCoordinate2D (c1, c2
#ifdef DEBUG_INFO
                                                     , startOfBlock.getType1 ()
                                                     , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                                                     ));

      ASSERT (index >= 0 && index < sizeGridValues);

      cudaCheckErrorCmd (cudaMemcpy (cpuGrid->getFieldPointValue (pos), &d_gridValues[index], sizeof (FieldPointValue), cudaMemcpyDeviceToHost));
    }
  }
} /* CudaGrid<GridCoordinate2D>::copyToCPU */

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate3D>::copyFromCPU (const GridCoordinate3D &start, /**< start coordinate of block to copy */
                                         const GridCoordinate3D &end) /**< end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;
  timeStep = cpuGrid->getTimeStep ();

  GridCoordinate3D zero (0, 0, 0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 (), startOfBlock.getType2 (), startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate3D startWithBuf = GridCoordinate3D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate3D endWithBuf = GridCoordinate3D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
    {
      grid_coord c2 = index2 - startWithBuf.get2 ();
      for (grid_coord index3 = startWithBuf.get3 (); index3 < endWithBuf.get3 (); ++index3)
      {
        grid_coord c3 = index3 - startWithBuf.get3 ();
        GridCoordinate3D pos = GridCoordinate3D (index1, index2, index3
#ifdef DEBUG_INFO
                                                 , startOfBlock.getType1 ()
                                                 , startOfBlock.getType2 ()
                                                 , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                                                 );
        grid_coord index = calculateIndexFromPosition (GridCoordinate3D (c1, c2, c3
#ifdef DEBUG_INFO
                                                       , startOfBlock.getType1 ()
                                                       , startOfBlock.getType2 ()
                                                       , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                                                       ));

        ASSERT (index >= 0 && index < sizeGridValues);

        cudaCheckErrorCmd (cudaMemcpy (&d_gridValues[index], cpuGrid->getFieldPointValue (pos), sizeof (FieldPointValue), cudaMemcpyHostToDevice));
      }
    }
  }
} /* CudaGrid<GridCoordinate3D>::copyFromCPU */

/**
 * Copy to CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate3D>::copyToCPU ()
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate3D zero (0, 0, 0
#ifdef DEBUG_INFO
                         , startOfBlock.getType1 (), startOfBlock.getType2 (), startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate3D startWithBuf = GridCoordinate3D::subWithBorder (startOfBlock, bufSize, zero);
  GridCoordinate3D endWithBuf = GridCoordinate3D::addWithBorder (endOfBlock, bufSize, cpuGrid->getSize ());

  for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
  {
    grid_coord c1 = index1 - startWithBuf.get1 ();
    for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
    {
      grid_coord c2 = index2 - startWithBuf.get2 ();
      for (grid_coord index3 = startWithBuf.get3 (); index3 < endWithBuf.get3 (); ++index3)
      {
        grid_coord c3 = index3 - startWithBuf.get3 ();
        GridCoordinate3D pos = GridCoordinate3D (index1, index2, index3
#ifdef DEBUG_INFO
                                                 , startOfBlock.getType1 ()
                                                 , startOfBlock.getType2 ()
                                                 , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                                                 );
        grid_coord index = calculateIndexFromPosition (GridCoordinate3D (c1, c2, c3
#ifdef DEBUG_INFO
                                                       , startOfBlock.getType1 ()
                                                       , startOfBlock.getType2 ()
                                                       , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                                                       ));

        ASSERT (index >= 0 && index < sizeGridValues);

        cudaCheckErrorCmd (cudaMemcpy (cpuGrid->getFieldPointValue (pos), &d_gridValues[index], sizeof (FieldPointValue), cudaMemcpyDeviceToHost));
      }
    }
  }
} /* CudaGrid<GridCoordinate3D>::copyToCPU */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate1D>::isLegitIndex (const GridCoordinate1D &position, /**< coordinate in grid */
                                          const GridCoordinate1D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  if (px >= sx)
  {
    return false;
  }

  return true;
} /* CudaGrid<GridCoordinate1D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from one-dimensional position
 *
 * @return one-dimensional coordinate from one-dimensional position
 */
template<>
CUDA_DEVICE CUDA_HOST
grid_coord
CudaGrid<GridCoordinate1D>::calculateIndexFromPosition (const GridCoordinate1D &position, /**< coordinate in grid */
                                                        const GridCoordinate1D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  return px;
} /* CudaGrid<GridCoordinate1D>::calculateIndexFromPosition */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate2D>::isLegitIndex (const GridCoordinate2D &position, /**< coordinate in grid */
                                          const GridCoordinate2D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  if (px >= sx)
  {
    return false;
  }
  else if (py >= sy)
  {
    return false;
  }

  return true;
} /* CudaGrid<GridCoordinate2D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from two-dimensional position
 *
 * @return one-dimensional coordinate from two-dimensional position
 */
template<>
CUDA_DEVICE CUDA_HOST
grid_coord
CudaGrid<GridCoordinate2D>::calculateIndexFromPosition (const GridCoordinate2D &position, /**< coordinate in grid */
                                                        const GridCoordinate2D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  return px * sy + py;
} /* CudaGrid<GridCoordinate2D>::calculateIndexFromPosition */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_HOST
bool
CudaGrid<GridCoordinate3D>::isLegitIndex (const GridCoordinate3D &position, /**< coordinate in grid */
                                          const GridCoordinate3D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  const grid_coord& pz = position.get3 ();
  const grid_coord& sz = sizeCoord.get3 ();

  if (px >= sx)
  {
    return false;
  }
  else if (py >= sy)
  {
    return false;
  }
  else if (pz >= sz)
  {
    return false;
  }

  return true;
} /* CudaGrid<GridCoordinate3D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from two-dimensional position
 *
 * @return one-dimensional coordinate from two-dimensional position
 */
template<>
CUDA_DEVICE CUDA_HOST
grid_coord
CudaGrid<GridCoordinate3D>::calculateIndexFromPosition (const GridCoordinate3D& position, /**< coordinate in grid */
                                                        const GridCoordinate3D& sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  const grid_coord& pz = position.get3 ();
  const grid_coord& sz = sizeCoord.get3 ();

  return px * sy * sz + py * sz + pz;
} /* CudaGrid<GridCoordinate3D>::calculateIndexFromPosition */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate1D
CudaGrid<GridCoordinate1D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  CoordinateType ct1 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
#endif /* !DEBUG_INFO */
  return GridCoordinate1D (index
#ifdef DEBUG_INFO
                           , ct1
#endif /* DEBUG_INFO */
                           );
} /* CudaGrid<GridCoordinate1D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate2D
CudaGrid<GridCoordinate2D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sy = getSize ().get2 ();

  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;

  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
#endif /* !DEBUG_INFO */

  return GridCoordinate2D (x, y
#ifdef DEBUG_INFO
                           , ct1, ct2
#endif /* DEBUG_INFO */
                           );
} /* CudaGrid<GridCoordinate2D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate3D
CudaGrid<GridCoordinate3D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sy = getSize ().get2 ();
  const grid_coord& sz = getSize ().get3 ();

  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;

  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
  ct3 = getSize ().getType3 ();
#endif /* !DEBUG_INFO */

  return GridCoordinate3D (x, y, z
#ifdef DEBUG_INFO
                           , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                           );
} /* CudaGrid<GridCoordinate3D>::calculatePositionFromIndex */

#endif /* CUDA_ENABLED */

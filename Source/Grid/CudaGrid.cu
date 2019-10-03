#define CUDA_SOURCES

#include "CudaGrid.h"

#ifdef CUDA_ENABLED

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getTotalPosition (const GridCoordinate1D & pos) const
{
  /*
   * This value is GridCoordinateSigned1D, but CudaGrid has no idea about GridCoordinateSigned1D,
   * that's why it can't be saved.
   */
  return GridCoordinate1D (startOfBlock.get1 () - bufSize.get1 () + pos.get1 ()
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                           );
}

#if defined (MODE_DIM2) || defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getTotalPosition (const GridCoordinate2D & pos) const
{
  /*
   * This value is GridCoordinateSigned2D, but CudaGrid has no idea about GridCoordinateSigned2D,
   * that's why it can't be saved.
   */
  return GridCoordinate2D (startOfBlock.get1 () - bufSize.get1 () + pos.get1 (),
                           startOfBlock.get2 () - bufSize.get2 () + pos.get2 ()
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
                           , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                           );
}

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getTotalPosition (const GridCoordinate3D & pos) const
{
  /*
   * This value is GridCoordinateSigned3D, but CudaGrid has no idea about GridCoordinateSigned3D,
   * that's why it can't be saved.
   */
  return GridCoordinate3D (startOfBlock.get1 () - bufSize.get1 () + pos.get1 (),
                           startOfBlock.get2 () - bufSize.get2 () + pos.get2 (),
                           startOfBlock.get3 () - bufSize.get3 () + pos.get3 ()
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
                           , startOfBlock.getType2 ()
                           , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                           );
}

#endif /* MODE_DIM3 */

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getRelativePosition (const GridCoordinate1D & pos) const
{
  /*
   * This value is GridCoordinateSigned1D, but CudaGrid has no idea about GridCoordinateSigned1D,
   * that's why it can't be saved.
   */
  return GridCoordinate1D (pos.get1 () - (startOfBlock.get1 () - bufSize.get1 ())
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
#endif /* DEBUG_INFO */
                           );
}

#if defined (MODE_DIM2) || defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getRelativePosition (const GridCoordinate2D & pos) const
{
  /*
   * This value is GridCoordinateSigned2D, but CudaGrid has no idea about GridCoordinateSigned2D,
   * that's why it can't be saved.
   */
  return GridCoordinate2D (pos.get1 () - (startOfBlock.get1 () - bufSize.get1 ()),
                           pos.get2 () - (startOfBlock.get2 () - bufSize.get2 ())
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
                           , startOfBlock.getType2 ()
#endif /* DEBUG_INFO */
                           );
}

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getRelativePosition (const GridCoordinate3D & pos) const
{
  /*
   * This value is GridCoordinateSigned3D, but CudaGrid has no idea about GridCoordinateSigned3D,
   * that's why it can't be saved.
   */
  return GridCoordinate3D (pos.get1 () - (startOfBlock.get1 () - bufSize.get1 ()),
                           pos.get2 () - (startOfBlock.get2 () - bufSize.get2 ()),
                           pos.get3 () - (startOfBlock.get3 () - bufSize.get3 ())
#ifdef DEBUG_INFO
                           , startOfBlock.getType1 ()
                           , startOfBlock.getType2 ()
                           , startOfBlock.getType3 ()
#endif /* DEBUG_INFO */
                           );
}

#endif /* MODE_DIM3 */

template <>
CUDA_DEVICE CUDA_HOST
bool
CudaGrid<GridCoordinate1D>::hasValueForCoordinate (const GridCoordinate1D & position) const
{
  if (!(position.get1 () >= startOfBlock.get1 () - hasLeft.get1 () * bufSize.get1 ())
      || !(position.get1 () < endOfBlock.get1 () + hasRight.get1 () * bufSize.get1 ()))
  {
    return false;
  }

  return true;
}

#if defined (MODE_DIM2) || defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
bool
CudaGrid<GridCoordinate2D>::hasValueForCoordinate (const GridCoordinate2D & position) const
{
  if (!(position.get1 () >= startOfBlock.get1 () - hasLeft.get1 () * bufSize.get1 ())
      || !(position.get2 () >= startOfBlock.get2 () - hasLeft.get2 () * bufSize.get2 ())
      || !(position.get1 () < endOfBlock.get1 () + hasRight.get1 () * bufSize.get1 ())
      || !(position.get2 () < endOfBlock.get2 () + hasRight.get2 () * bufSize.get2 ()))
  {
    return false;
  }

  return true;
}

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

template <>
CUDA_DEVICE CUDA_HOST
bool
CudaGrid<GridCoordinate3D>::hasValueForCoordinate (const GridCoordinate3D & position) const
{
  if (!(position.get1 () >= startOfBlock.get1 () - hasLeft.get1 () * bufSize.get1 ())
      || !(position.get2 () >= startOfBlock.get2 () - hasLeft.get2 () * bufSize.get2 ())
      || !(position.get3 () >= startOfBlock.get3 () - hasLeft.get3 () * bufSize.get3 ())
      || !(position.get1 () < endOfBlock.get1 () + hasRight.get1 () * bufSize.get1 ())
      || !(position.get2 () < endOfBlock.get2 () + hasRight.get2 () * bufSize.get2 ())
      || !(position.get3 () < endOfBlock.get3 () + hasRight.get3 () * bufSize.get3 ()))
  {
    return false;
  }

  return true;
}

#endif /* MODE_DIM3 */

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

#if defined (MODE_DIM2) || defined (MODE_DIM3)

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

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

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

#endif /* MODE_DIM3 */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getComputationStart (const GridCoordinate1D & diffPosStart) const /**< offset from the left border */
{
  grid_coord c1 = getShareStep () + 1;

  if (hasLeft.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosStart.get1 ();
  }

  return GridCoordinate1D (c1
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate1D>::getComputationStart */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getComputationStart (const GridCoordinate2D & diffPosStart) const /**< offset from the left border */
{
  grid_coord c1 = getShareStep () + 1;
  grid_coord c2 = getShareStep () + 1;

  if (hasLeft.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosStart.get1 ();
  }

  if (hasLeft.get2 () == 0)
  {
    c2 = bufSize.get2 () + diffPosStart.get2 ();
  }

  return GridCoordinate2D (c1, c2
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
                           , getSize ().getType2 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate2D>::getComputationStart */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getComputationStart (const GridCoordinate3D & diffPosStart) const /**< offset from the left border */
{
  grid_coord c1 = getShareStep () + 1;
  grid_coord c2 = getShareStep () + 1;
  grid_coord c3 = getShareStep () + 1;

  if (hasLeft.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosStart.get1 ();
  }

  if (hasLeft.get2 () == 0)
  {
    c2 = bufSize.get2 () + diffPosStart.get2 ();
  }

  if (hasLeft.get3 () == 0)
  {
    c3 = bufSize.get3 () + diffPosStart.get3 ();
  }

  return GridCoordinate3D (c1, c2, c3
#ifdef DEBUG_INFO
                           , getSize ().getType1 ()
                           , getSize ().getType2 ()
                           , getSize ().getType3 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate3D>::getComputationStart */

#endif /* MODE_DIM3 */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate1D
CudaGrid<GridCoordinate1D>::getComputationEnd (const GridCoordinate1D & diffPosEnd) const
{
  grid_coord c1 = getShareStep () + 1;

  if (hasRight.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosEnd.get1 ();
  }

  return getSize () - GridCoordinate1D (c1
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate1D>::getComputationEnd () */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate2D
CudaGrid<GridCoordinate2D>::getComputationEnd (const GridCoordinate2D & diffPosEnd) const
{
  grid_coord c1 = getShareStep () + 1;
  grid_coord c2 = getShareStep () + 1;

  if (hasRight.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosEnd.get1 ();
  }

  if (hasRight.get2 () == 0)
  {
    c2 = bufSize.get2 () + diffPosEnd.get2 ();
  }

  return getSize () - GridCoordinate2D (c1, c2
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate2D>::getComputationEnd () */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <>
CUDA_DEVICE CUDA_HOST
GridCoordinate3D
CudaGrid<GridCoordinate3D>::getComputationEnd (const GridCoordinate3D & diffPosEnd) const
{
  grid_coord c1 = getShareStep () + 1;
  grid_coord c2 = getShareStep () + 1;
  grid_coord c3 = getShareStep () + 1;

  if (hasRight.get1 () == 0)
  {
    c1 = bufSize.get1 () + diffPosEnd.get1 ();
  }

  if (hasRight.get2 () == 0)
  {
    c2 = bufSize.get2 () + diffPosEnd.get2 ();
  }

  if (hasRight.get3 () == 0)
  {
    c3 = bufSize.get3 () + diffPosEnd.get3 ();
  }

  return getSize () - GridCoordinate3D (c1, c2, c3
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
                                        , getSize ().getType3 ()
#endif /* DEBUG_INFO */
  );
} /* CudaGrid<GridCoordinate3D>::getComputationEnd () */

#endif /* MODE_DIM3 */

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate1D>::copyFromCPU (const GridCoordinate1D &start, /**< absolute start coordinate of block to copy */
                                         const GridCoordinate1D &end) /**< absolute end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;

  hasLeft = GRID_COORDINATE_1D (1, start.getType1 ());
  hasRight = GRID_COORDINATE_1D (1, start.getType1 ());

  if (start.get1 () == grid_coord (0))
  {
    hasLeft.set1 (0);
  }
  if (end.get1 () == cpuGrid->getTotalSize ().get1 ())
  {
    hasRight.set1 (0);
  }

  GridCoordinate1D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate1D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned1D startWithBuf (relStart.get1 () - bufSize.get1 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned1D endWithBuf (relEnd.get1 () + bufSize.get1 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
#endif /* DEBUG_INFO */
                                     );

  // grid_coord count = endWithBuf.get1 () - startWithBuf.get1 ();
  // ASSERT (cpuGrid->getSize ().get1 () <= count);
  // memcpy (helperGridValues, cpuGrid->getRaw () + cpuGrid->calculateIndexFromPosition (startWithBuf), count * sizeof (FieldPointValue));

  for (int t = 0; t < storedSteps; ++t)
  {
    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      GridCoordinate1D pos = GRID_COORDINATE_1D (index1, startOfBlock.getType1 ());
      grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_1D (c1, startOfBlock.getType1 ()));

      ASSERT (index >= 0 && index < sizeGridValues);

      /*
       * Copy to helper buffer first
       */
      helperGridValues[index] = *cpuGrid->getFieldValue (pos, t);
    }

    cudaCheckErrorCmd (cudaMemcpy (gridValuesDevicePointers[t], helperGridValues, sizeGridValues * sizeof (FieldValue), cudaMemcpyHostToDevice));
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
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate1D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate1D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned1D startWithBuf (relStart.get1 () - bufSize.get1 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned1D endWithBuf (relEnd.get1 () + bufSize.get1 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
#endif /* DEBUG_INFO */
                                     );

  // grid_coord count = endWithBuf.get1 () - startWithBuf.get1 ();
  // ASSERT (cpuGrid->getSize ().get1 () <= count);
  // memcpy (cpuGrid->getRaw () + cpuGrid->calculateIndexFromPosition (startWithBuf), helperGridValues, count * sizeof (FieldPointValue));

  for (int t = 0; t < storedSteps; ++t)
  {
    cudaCheckErrorCmd (cudaMemcpy (helperGridValues, gridValuesDevicePointers[t], sizeGridValues * sizeof (FieldValue), cudaMemcpyDeviceToHost));

    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      GridCoordinate1D pos = GRID_COORDINATE_1D (index1, startOfBlock.getType1 ());
      grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_1D (c1, startOfBlock.getType1 ()));

      ASSERT (index >= 0 && index < sizeGridValues);

      /*
       * Copy to helper buffer first
       */
      cpuGrid->setFieldValue (helperGridValues[index], pos, t);
    }
  }
} /* CudaGrid<GridCoordinate1D>::copyToCPU */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate2D>::copyFromCPU (const GridCoordinate2D &start, /**< absolute start coordinate of block to copy */
                                         const GridCoordinate2D &end) /**< absolute end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;

  hasLeft = GRID_COORDINATE_2D (1, 1, start.getType1 (), start.getType2 ());
  hasRight = GRID_COORDINATE_2D (1, 1, start.getType1 (), start.getType2 ());

  if (start.get1 () == grid_coord (0))
  {
    hasLeft.set1 (0);
  }
  if (start.get2 () == grid_coord (0))
  {
    hasLeft.set2 (0);
  }
  if (end.get1 () == cpuGrid->getTotalSize ().get1 ())
  {
    hasRight.set1 (0);
  }
  if (end.get2 () == cpuGrid->getTotalSize ().get2 ())
  {
    hasRight.set2 (0);
  }

  GridCoordinate2D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate2D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned2D startWithBuf (relStart.get1 () - bufSize.get1 (),
                                       relStart.get2 () - bufSize.get2 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
                                       , relStart.getType2 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned2D endWithBuf (relEnd.get1 () + bufSize.get1 (),
                                     relEnd.get2 () + bufSize.get2 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
                                     , relEnd.getType2 ()
#endif /* DEBUG_INFO */
                                     );

  for (int t = 0; t < storedSteps; ++t)
  {
    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
      {
        if (index2 < 0 || index2 >= cpuGrid->getSize ().get2 ())
        {
          continue;
        }

        grid_coord c2 = index2 - startWithBuf.get2 ();

        GridCoordinate2D pos = GRID_COORDINATE_2D (index1, index2, startOfBlock.getType1 (), startOfBlock.getType2 ());
        grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_2D (c1, c2, startOfBlock.getType1 (), startOfBlock.getType2 ()));

        ASSERT (index >= 0 && index < sizeGridValues);

        /*
         * Copy to helper buffer first
         */
        helperGridValues[index] = *cpuGrid->getFieldValue (pos, t);
      }
    }

    cudaCheckErrorCmd (cudaMemcpy (gridValuesDevicePointers[t], helperGridValues, sizeGridValues * sizeof (FieldValue), cudaMemcpyHostToDevice));
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
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate2D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate2D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned2D startWithBuf (relStart.get1 () - bufSize.get1 (),
                                       relStart.get2 () - bufSize.get2 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
                                       , relStart.getType2 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned2D endWithBuf (relEnd.get1 () + bufSize.get1 (),
                                     relEnd.get2 () + bufSize.get2 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
                                     , relEnd.getType2 ()
#endif /* DEBUG_INFO */
                                     );

  for (int t = 0; t < storedSteps; ++t)
  {
    cudaCheckErrorCmd (cudaMemcpy (helperGridValues, gridValuesDevicePointers[t], sizeGridValues * sizeof (FieldValue), cudaMemcpyDeviceToHost));

    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
      {
        if (index2 < 0 || index2 >= cpuGrid->getSize ().get2 ())
        {
          continue;
        }

        grid_coord c2 = index2 - startWithBuf.get2 ();

        GridCoordinate2D pos = GRID_COORDINATE_2D (index1, index2,
                                                   startOfBlock.getType1 (),
                                                   startOfBlock.getType2 ());
        grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_2D (c1, c2,
                                                                           startOfBlock.getType1 (),
                                                                           startOfBlock.getType2 ()));

        ASSERT (index >= 0 && index < sizeGridValues);

        /*
         * Copy to helper buffer first
         */
        cpuGrid->setFieldValue (helperGridValues[index], pos, t);
      }
    }
  }
} /* CudaGrid<GridCoordinate2D>::copyToCPU */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Copy from CPU grid
 */
template <>
CUDA_HOST
void
CudaGrid<GridCoordinate3D>::copyFromCPU (const GridCoordinate3D &start, /**< absolute start coordinate of block to copy */
                                         const GridCoordinate3D &end) /**< absolute end coordinate of block to copy */
{
  ASSERT (cpuGrid != NULLPTR);
  ASSERT (d_gridValues != NULLPTR);
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);

  startOfBlock = start;
  endOfBlock = end;

  hasLeft = GRID_COORDINATE_3D (1, 1, 1, start.getType1 (), start.getType2 (), start.getType3 ());
  hasRight = GRID_COORDINATE_3D (1, 1, 1, start.getType1 (), start.getType2 (), start.getType3 ());

  if (start.get1 () == grid_coord (0))
  {
    hasLeft.set1 (0);
  }
  if (start.get2 () == grid_coord (0))
  {
    hasLeft.set2 (0);
  }
  if (start.get3 () == grid_coord (0))
  {
    hasLeft.set3 (0);
  }
  if (end.get1 () == cpuGrid->getTotalSize ().get1 ())
  {
    hasRight.set1 (0);
  }
  if (end.get2 () == cpuGrid->getTotalSize ().get2 ())
  {
    hasRight.set2 (0);
  }
  if (end.get3 () == cpuGrid->getTotalSize ().get3 ())
  {
    hasRight.set3 (0);
  }

  GridCoordinate3D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate3D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned3D startWithBuf (relStart.get1 () - bufSize.get1 (),
                                       relStart.get2 () - bufSize.get2 (),
                                       relStart.get3 () - bufSize.get3 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
                                       , relStart.getType2 ()
                                       , relStart.getType3 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned3D endWithBuf (relEnd.get1 () + bufSize.get1 (),
                                     relEnd.get2 () + bufSize.get2 (),
                                     relEnd.get3 () + bufSize.get3 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
                                     , relEnd.getType2 ()
                                     , relEnd.getType3 ()
#endif /* DEBUG_INFO */
                                     );

  for (int t = 0; t < storedSteps; ++t)
  {
    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
      {
        if (index2 < 0 || index2 >= cpuGrid->getSize ().get2 ())
        {
          continue;
        }

        grid_coord c2 = index2 - startWithBuf.get2 ();

        for (grid_coord index3 = startWithBuf.get3 (); index3 < endWithBuf.get3 (); ++index3)
        {
          if (index3 < 0 || index3 >= cpuGrid->getSize ().get3 ())
          {
            continue;
          }

          grid_coord c3 = index3 - startWithBuf.get3 ();

          GridCoordinate3D pos = GRID_COORDINATE_3D (index1, index2, index3,
                                                     startOfBlock.getType1 (),
                                                     startOfBlock.getType2 (),
                                                     startOfBlock.getType3 ());
          grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_3D (c1, c2, c3,
                                                         startOfBlock.getType1 (),
                                                         startOfBlock.getType2 (),
                                                         startOfBlock.getType3 ()));

          ASSERT (index >= 0 && index < sizeGridValues);

          /*
           * Copy to helper buffer first
           */
          helperGridValues[index] = *cpuGrid->getFieldValue (pos, t);
        }
      }
    }

    cudaCheckErrorCmd (cudaMemcpy (gridValuesDevicePointers[t], helperGridValues, sizeGridValues * sizeof (FieldValue), cudaMemcpyHostToDevice));
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
  ASSERT (gridValuesDevicePointers != NULLPTR);
  ASSERT (helperGridValues != NULLPTR);
  ASSERT (startOfBlock != endOfBlock);

  GridCoordinate3D relStart = cpuGrid->getRelativePosition (startOfBlock);
  GridCoordinate3D relEnd = cpuGrid->getRelativePosition (endOfBlock);

  GridCoordinateSigned3D startWithBuf (relStart.get1 () - bufSize.get1 (),
                                       relStart.get2 () - bufSize.get2 (),
                                       relStart.get3 () - bufSize.get3 ()
#ifdef DEBUG_INFO
                                       , relStart.getType1 ()
                                       , relStart.getType2 ()
                                       , relStart.getType3 ()
#endif /* DEBUG_INFO */
                                       );
  GridCoordinateSigned3D endWithBuf (relEnd.get1 () + bufSize.get1 (),
                                     relEnd.get2 () + bufSize.get2 (),
                                     relEnd.get3 () + bufSize.get3 ()
#ifdef DEBUG_INFO
                                     , relEnd.getType1 ()
                                     , relEnd.getType2 ()
                                     , relEnd.getType3 ()
#endif /* DEBUG_INFO */
                                     );

  for (int t = 0; t < storedSteps; ++t)
  {
    cudaCheckErrorCmd (cudaMemcpy (helperGridValues, gridValuesDevicePointers[t], sizeGridValues * sizeof (FieldValue), cudaMemcpyDeviceToHost));

    for (grid_coord index1 = startWithBuf.get1 (); index1 < endWithBuf.get1 (); ++index1)
    {
      if (index1 < 0 || index1 >= cpuGrid->getSize ().get1 ())
      {
        continue;
      }

      grid_coord c1 = index1 - startWithBuf.get1 ();

      for (grid_coord index2 = startWithBuf.get2 (); index2 < endWithBuf.get2 (); ++index2)
      {
        if (index2 < 0 || index2 >= cpuGrid->getSize ().get2 ())
        {
          continue;
        }

        grid_coord c2 = index2 - startWithBuf.get2 ();

        for (grid_coord index3 = startWithBuf.get3 (); index3 < endWithBuf.get3 (); ++index3)
        {
          if (index3 < 0 || index3 >= cpuGrid->getSize ().get3 ())
          {
            continue;
          }

          grid_coord c3 = index3 - startWithBuf.get3 ();

          GridCoordinate3D pos = GRID_COORDINATE_3D (index1, index2, index3,
                                                     startOfBlock.getType1 (),
                                                     startOfBlock.getType2 (),
                                                     startOfBlock.getType3 ());
          grid_coord index = calculateIndexFromPosition (GRID_COORDINATE_3D (c1, c2, c3,
                                                         startOfBlock.getType1 (),
                                                         startOfBlock.getType2 (),
                                                         startOfBlock.getType3 ()));

          ASSERT (index >= 0 && index < sizeGridValues);

          /*
           * Copy to helper buffer first
           */
          cpuGrid->setFieldValue (helperGridValues[index], pos, t);
        }
      }
    }
  }
} /* CudaGrid<GridCoordinate3D>::copyToCPU */

#endif /* MODE_DIM3 */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_DEVICE CUDA_HOST
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

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_DEVICE CUDA_HOST
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

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
CUDA_DEVICE CUDA_HOST
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

#endif /* MODE_DIM3 */

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

#if defined (MODE_DIM2) || defined (MODE_DIM3)

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

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

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

#endif /* MODE_DIM3 */

#endif /* CUDA_ENABLED */

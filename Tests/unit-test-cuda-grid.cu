/*
 * Unit test for basic operations with CudaGrid
 */
#define CUDA_SOURCES

#include <iostream>

#include "Assert.h"
#include "GridCoordinate3D.h"
#include "CudaGrid.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate1D> *grid)
{
  GridCoordinate1D pos ((blockIdx.x * blockDim.x) + threadIdx.x
#ifdef DEBUG_INFO
                        , grid->getSize ().getType1 ()
#endif /* DEBUG_INFO */
                       );

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldPointValue *val = grid->getFieldPointValue (pos);
  val->shiftInTime ();
  grid_coord index = pos.calculateTotalCoord ();
  val->setCurValue (FIELDVALUE (index * 23, index * 17));

  PROGRAM_OK_EXIT;
}

__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate2D> *grid)
{
  GridCoordinate2D pos ((blockIdx.x * blockDim.x) + threadIdx.x,
                        (blockIdx.y * blockDim.y) + threadIdx.y
#ifdef DEBUG_INFO
                        , grid->getSize ().getType1 ()
                        , grid->getSize ().getType2 ()
#endif /* DEBUG_INFO */
                       );

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldPointValue *val = grid->getFieldPointValue (pos);
  val->shiftInTime ();
  grid_coord index = pos.calculateTotalCoord ();
  val->setCurValue (FIELDVALUE (index * 23, index * 17));

  PROGRAM_OK_EXIT;
}

__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate3D> *grid)
{
  GridCoordinate3D pos ((blockIdx.x * blockDim.x) + threadIdx.x,
                        (blockIdx.y * blockDim.y) + threadIdx.y,
                        (blockIdx.z * blockDim.z) + threadIdx.z
#ifdef DEBUG_INFO
                        , grid->getSize ().getType1 ()
                        , grid->getSize ().getType2 ()
                        , grid->getSize ().getType3 ()
#endif /* DEBUG_INFO */
                       );

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldPointValue *val = grid->getFieldPointValue (pos);
  val->shiftInTime ();
  grid_coord index = pos.calculateTotalCoord ();
  val->setCurValue (FIELDVALUE (index * 23, index * 17));

  PROGRAM_OK_EXIT;
}

void testFunc1D (GridCoordinate1D overallSize, GridCoordinate1D bufSize)
{
  Grid<GridCoordinate1D> cpuGrid (overallSize, 0);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate1D zero (0
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate1D one (1
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
#endif /* DEBUG_INFO */
                        );

  /*
   * CudaGrid with capacity for whole cpu grid
   */
  CudaGrid<GridCoordinate1D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  ASSERT (cudaGrid.getBufSize () == bufSize);
  ASSERT (cudaGrid.getSizeGridValues () == (overallSize + bufSize * 2).calculateTotalCoord ());
  ASSERT (cudaGrid.getShareStep () == 0);
  ASSERT (cudaGrid.getTotalSize () == overallSize);
  ASSERT (cudaGrid.getTotalPosition (bufSize) == zero);
  ASSERT (cudaGrid.getRelativePosition (zero) == bufSize);
  ASSERT (cudaGrid.hasValueForCoordinate (zero));
  ASSERT (cudaGrid.getFieldPointValueByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, 1, 1);
  dim3 threads (2, 1, 1);

  CudaGrid<GridCoordinate1D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate1D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , overallSize.getType1 ()
#endif /* DEBUG_INFO */
                         );
    FieldPointValue *val = cpuGrid.getFieldPointValue (pos - bufSize);
    grid_coord index = pos.calculateTotalCoord ();

    ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
    ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
  }

  {
    CudaGrid<GridCoordinate1D> cudaGrid2 (overallSize / 2, bufSize, &cpuGrid);

    cudaGrid2.copyFromCPU (zero, overallSize / 2);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == zero);
    ASSERT (cudaGrid2.getRelativePosition (zero) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (overallSize / 2 + bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == overallSize / 2 + bufSize);
    ASSERT (cudaGrid2.hasValueForCoordinate (zero));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one + bufSize);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize * 2 - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 1);

    cudaGrid2.copyFromCPU (overallSize / 2, overallSize);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (zero) == overallSize / 2 - bufSize);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2 - bufSize) == zero);
    ASSERT (cudaGrid2.hasValueForCoordinate (overallSize / 2));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
  }
}

void testFunc2D (GridCoordinate2D overallSize, GridCoordinate2D bufSize)
{
  Grid<GridCoordinate2D> cpuGrid (overallSize, 0);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate2D zero (0, 0
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
                         , overallSize.getType2 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate2D one (1, 1
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
                         , overallSize.getType2 ()
#endif /* DEBUG_INFO */
                        );

  /*
   * CudaGrid with capacity for all cpu grid
   */
  CudaGrid<GridCoordinate2D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  ASSERT (cudaGrid.getBufSize () == bufSize);
  ASSERT (cudaGrid.getSizeGridValues () == (overallSize + bufSize * 2).calculateTotalCoord ());
  ASSERT (cudaGrid.getShareStep () == 0);
  ASSERT (cudaGrid.getTotalSize () == overallSize);
  ASSERT (cudaGrid.getTotalPosition (bufSize) == zero);
  ASSERT (cudaGrid.getRelativePosition (zero) == bufSize);
  ASSERT (cudaGrid.hasValueForCoordinate (zero));
  ASSERT (cudaGrid.getFieldPointValueByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasLeft ().get2 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get2 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, cudaGrid.getSize ().get2 () / 2, 1);
  dim3 threads (2, 2, 1);

  CudaGrid<GridCoordinate2D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate2D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate2D>), cudaMemcpyHostToDevice));
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    for (grid_coord j = bufSize.get2 (); j < cudaGrid.getSize ().get2 () - bufSize.get2 (); ++j)
    {
      GridCoordinate2D pos (i, j
#ifdef DEBUG_INFO
                            , overallSize.getType1 ()
                            , overallSize.getType2 ()
#endif /* DEBUG_INFO */
                           );
      FieldPointValue *val = cpuGrid.getFieldPointValue (pos - bufSize);
      grid_coord index = pos.calculateTotalCoord ();

      ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
      ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
    }
  }

  {
    CudaGrid<GridCoordinate2D> cudaGrid2 (overallSize / 2, bufSize, &cpuGrid);

    cudaGrid2.copyFromCPU (zero, overallSize / 2);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == zero);
    ASSERT (cudaGrid2.getRelativePosition (zero) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (overallSize / 2 + bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == overallSize / 2 + bufSize);
    ASSERT (cudaGrid2.hasValueForCoordinate (zero));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one + bufSize);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize * 2 - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 0);
    ASSERT (cudaGrid2.getHasLeft ().get2 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get2 () == 1);

    cudaGrid2.copyFromCPU (overallSize / 2, overallSize);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (zero) == overallSize / 2 - bufSize);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2 - bufSize) == zero);
    ASSERT (cudaGrid2.hasValueForCoordinate (overallSize / 2));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasLeft ().get2 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get2 () == 0);
  }
}

void testFunc3D (GridCoordinate3D overallSize, GridCoordinate3D bufSize)
{
  Grid<GridCoordinate3D> cpuGrid (overallSize, 0);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate3D zero (0, 0, 0
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
                         , overallSize.getType2 ()
                         , overallSize.getType3 ()
#endif /* DEBUG_INFO */
                        );
  GridCoordinate3D one (1, 1, 1
#ifdef DEBUG_INFO
                         , overallSize.getType1 ()
                         , overallSize.getType2 ()
                         , overallSize.getType3 ()
#endif /* DEBUG_INFO */
                        );

  /*
   * CudaGrid with capacity for all cpu grid
   */
  CudaGrid<GridCoordinate3D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  ASSERT (cudaGrid.getBufSize () == bufSize);
  ASSERT (cudaGrid.getSizeGridValues () == (overallSize + bufSize * 2).calculateTotalCoord ());
  ASSERT (cudaGrid.getShareStep () == 0);
  ASSERT (cudaGrid.getTotalSize () == overallSize);
  ASSERT (cudaGrid.getTotalPosition (bufSize) == zero);
  ASSERT (cudaGrid.getRelativePosition (zero) == bufSize);
  ASSERT (cudaGrid.hasValueForCoordinate (zero));
  ASSERT (cudaGrid.getFieldPointValueByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid.getFieldPointValue (bufSize));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasLeft ().get2 () == 0);
  ASSERT (cudaGrid.getHasLeft ().get3 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get2 () == 0);
  ASSERT (cudaGrid.getHasRight ().get3 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, cudaGrid.getSize ().get2 () / 2, cudaGrid.getSize ().get3 () / 2);
  dim3 threads (2, 2, 2);

  CudaGrid<GridCoordinate3D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate3D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate3D>), cudaMemcpyHostToDevice));
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    for (grid_coord j = bufSize.get2 (); j < cudaGrid.getSize ().get2 () - bufSize.get2 (); ++j)
    {
      for (grid_coord k = bufSize.get3 (); k < cudaGrid.getSize ().get3 () - bufSize.get3 (); ++k)
      {
        GridCoordinate3D pos (i, j, k
#ifdef DEBUG_INFO
                              , overallSize.getType1 ()
                              , overallSize.getType2 ()
                              , overallSize.getType3 ()
#endif /* DEBUG_INFO */
                             );
        FieldPointValue *val = cpuGrid.getFieldPointValue (pos - bufSize);
        grid_coord index = pos.calculateTotalCoord ();

        ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
        ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
      }
    }
  }

  {
    CudaGrid<GridCoordinate3D> cudaGrid2 (overallSize / 2, bufSize, &cpuGrid);

    cudaGrid2.copyFromCPU (zero, overallSize / 2);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == zero);
    ASSERT (cudaGrid2.getRelativePosition (zero) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (overallSize / 2 + bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == overallSize / 2 + bufSize);
    ASSERT (cudaGrid2.hasValueForCoordinate (zero));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (zero) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one + bufSize);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize * 2 - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 0);
    ASSERT (cudaGrid2.getHasLeft ().get2 () == 0);
    ASSERT (cudaGrid2.getHasLeft ().get3 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get2 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get3 () == 1);

    cudaGrid2.copyFromCPU (overallSize / 2, overallSize);
    ASSERT (cudaGrid2.getBufSize () == bufSize);
    ASSERT (cudaGrid2.getSizeGridValues () == (overallSize / 2 + bufSize * 2).calculateTotalCoord ());
    ASSERT (cudaGrid2.getShareStep () == 0);
    ASSERT (cudaGrid2.getTotalSize () == overallSize);
    ASSERT (cudaGrid2.getTotalPosition (bufSize) == overallSize / 2);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2) == bufSize);
    ASSERT (cudaGrid2.getTotalPosition (zero) == overallSize / 2 - bufSize);
    ASSERT (cudaGrid2.getRelativePosition (overallSize / 2 - bufSize) == zero);
    ASSERT (cudaGrid2.hasValueForCoordinate (overallSize / 2));
    ASSERT (cudaGrid2.getFieldPointValueByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getFieldPointValueOrNullByAbsolutePos (overallSize / 2) == cudaGrid2.getFieldPointValue (bufSize));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasLeft ().get2 () == 1);
    ASSERT (cudaGrid2.getHasLeft ().get3 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get2 () == 0);
    ASSERT (cudaGrid2.getHasRight ().get3 () == 0);
  }
}

int main (int argc, char** argv)
{
  cudaCheckErrorCmd (cudaSetDevice(STOI (argv[1])));
  solverSettings.Initialize ();

  int gridSizeX = 32;
  int gridSizeY = 46;
  int gridSizeZ = 40;

  testFunc1D (GridCoordinate1D (gridSizeX, CoordinateType::X),
              GridCoordinate1D (1, CoordinateType::X));
  testFunc1D (GridCoordinate1D (gridSizeY, CoordinateType::Y),
              GridCoordinate1D (1, CoordinateType::Y));
  testFunc1D (GridCoordinate1D (gridSizeZ, CoordinateType::Z),
              GridCoordinate1D (1, CoordinateType::Z));

  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y),
              GridCoordinate2D (1, 1, CoordinateType::X, CoordinateType::Y));
  testFunc2D (GridCoordinate2D (gridSizeY, gridSizeZ, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate2D (1, 1, CoordinateType::Y, CoordinateType::Z));
  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeZ, CoordinateType::X, CoordinateType::Z),
              GridCoordinate2D (1, 1, CoordinateType::X, CoordinateType::Z));

  testFunc3D (GridCoordinate3D (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate3D (1, 1, 1, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));

  solverSettings.Uninitialize ();

  return 0;
} /* main */

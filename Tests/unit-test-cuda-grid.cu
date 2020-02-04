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

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

#if defined (MODE_DIM1)
__global__ void shiftInTime (CudaExitStatus *retval,
                             CudaGrid<GridCoordinate1D> *grid)
{
  grid->shiftInTime ();

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)
__global__ void shiftInTime (CudaExitStatus *retval,
                             CudaGrid<GridCoordinate2D> *grid)
{
  grid->shiftInTime ();

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)
__global__ void shiftInTime (CudaExitStatus *retval,
                             CudaGrid<GridCoordinate3D> *grid)
{
  grid->shiftInTime ();

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM3 */

#if defined (MODE_DIM1)
__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate1D> *grid)
{
  GridCoordinate1D pos = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                             grid->getSize ().getType1 ());

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldValue *val = grid->getFieldValue (pos, 1);
  grid_coord index = grid->calculateIndexFromPosition (pos);
  grid->setFieldValue (*val + FIELDVALUE (index * 23, index * 17), index, 0);

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)
__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate2D> *grid)
{
  GridCoordinate2D pos = GRID_COORDINATE_2D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                             (blockIdx.y * blockDim.y) + threadIdx.y,
                                             grid->getSize ().getType1 (),
                                             grid->getSize ().getType2 ());

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldValue *val = grid->getFieldValue (pos, 1);
  grid_coord index = grid->calculateIndexFromPosition (pos);
  grid->setFieldValue (*val + FIELDVALUE (index * 23, index * 17), index, 0);

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)
__global__ void cudaCalculate (CudaExitStatus *retval,
                               CudaGrid<GridCoordinate3D> *grid)
{
  GridCoordinate3D pos = GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                             (blockIdx.y * blockDim.y) + threadIdx.y,
                                             (blockIdx.z * blockDim.z) + threadIdx.z,
                                             grid->getSize ().getType1 (),
                                             grid->getSize ().getType2 (),
                                             grid->getSize ().getType3 ());

  ALWAYS_ASSERT (pos < grid->getSize ());

  FieldValue *val = grid->getFieldValue (pos, 1);
  grid_coord index = grid->calculateIndexFromPosition (pos);
  grid->setFieldValue (*val + FIELDVALUE (index * 23, index * 17), index, 0);

  PROGRAM_OK_EXIT;
}
#endif /* MODE_DIM3 */

#if defined (MODE_DIM1)
void testFunc1D (GridCoordinate1D overallSize, GridCoordinate1D bufSize)
{
  Grid<GridCoordinate1D> cpuGrid (overallSize, 2);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate1D zero = overallSize.getZero ();
  GridCoordinate1D one = GRID_COORDINATE_1D (1, overallSize.getType1 ());

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
  ASSERT (cudaGrid.getFieldValueByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  CudaGrid<GridCoordinate1D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate1D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));

  dim3 blocks_shift (1, 1, 1);
  dim3 threads_shift (1, 1, 1);
  cudaCheckExitStatus (shiftInTime <<< blocks_shift, threads_shift >>> (exitStatusCuda, d_cudaGrid));
  cudaGrid.shiftInTime ();

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, 1, 1);
  dim3 threads (2, 1, 1);
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, overallSize.getType1 ());
    grid_coord index = cudaGrid.calculateIndexFromPosition (pos);

    GridCoordinate1D posCPU = GRID_COORDINATE_1D (i - bufSize.get1 (), overallSize.getType1 ());
    grid_coord coord = cpuGrid.calculateIndexFromPosition (posCPU);

    ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 0) == FIELDVALUE (17, 1022) + FIELDVALUE (index * 23, index * 17));
    ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 1) == FIELDVALUE (17, 1022));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
  }
}
#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)
void testFunc2D (GridCoordinate2D overallSize, GridCoordinate2D bufSize)
{
  Grid<GridCoordinate2D> cpuGrid (overallSize, 2);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate2D zero = GRID_COORDINATE_2D (0, 0, overallSize.getType1 (), overallSize.getType2 ());
  GridCoordinate2D one = GRID_COORDINATE_2D (1, 1, overallSize.getType1 (), overallSize.getType2 ());

  /*
   * CudaGrid with capacity for whole cpu grid
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
  ASSERT (cudaGrid.getFieldValueByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  CudaGrid<GridCoordinate2D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate2D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate2D>), cudaMemcpyHostToDevice));

  dim3 blocks_shift (1, 1, 1);
  dim3 threads_shift (1, 1, 1);
  cudaCheckExitStatus (shiftInTime <<< blocks_shift, threads_shift >>> (exitStatusCuda, d_cudaGrid));
  cudaGrid.shiftInTime ();

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, cudaGrid.getSize ().get2 () / 2, 1);
  dim3 threads (2, 2, 1);
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    for (grid_coord j = bufSize.get2 (); j < cudaGrid.getSize ().get2 () - bufSize.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j,
                                                 overallSize.getType1 (),
                                                 overallSize.getType2 ());
      grid_coord index = cudaGrid.calculateIndexFromPosition (pos);

      GridCoordinate2D posCPU = GRID_COORDINATE_2D (i - bufSize.get1 (), j - bufSize.get2 (),
                                                    overallSize.getType1 (),
                                                    overallSize.getType2 ());
      grid_coord coord = cpuGrid.calculateIndexFromPosition (posCPU);

      ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 0) == FIELDVALUE (17, 1022) + FIELDVALUE (index * 23, index * 17));
      ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 1) == FIELDVALUE (17, 1022));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
  }
}
#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)
void testFunc3D (GridCoordinate3D overallSize, GridCoordinate3D bufSize)
{
  Grid<GridCoordinate3D> cpuGrid (overallSize, 2);
  cpuGrid.initialize (FIELDVALUE (17, 1022));

  GridCoordinate3D zero = GRID_COORDINATE_3D (0, 0, 0, overallSize.getType1 (), overallSize.getType2 (), overallSize.getType3 ());
  GridCoordinate3D one = GRID_COORDINATE_3D (1, 1, 1, overallSize.getType1 (), overallSize.getType2 (), overallSize.getType3 ());

  /*
   * CudaGrid with capacity for whole cpu grid
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
  ASSERT (cudaGrid.getFieldValueByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid.getFieldValue (bufSize, 0));
  ASSERT (cudaGrid.getComputationStart (one) == one + bufSize);
  ASSERT (cudaGrid.getComputationEnd (one) == overallSize + bufSize - one);
  ASSERT (cudaGrid.getHasLeft ().get1 () == 0);
  ASSERT (cudaGrid.getHasRight ().get1 () == 0);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  CudaGrid<GridCoordinate3D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate3D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate3D>), cudaMemcpyHostToDevice));

  dim3 blocks_shift (1, 1, 1);
  dim3 threads_shift (1, 1, 1);
  cudaCheckExitStatus (shiftInTime <<< blocks_shift, threads_shift >>> (exitStatusCuda, d_cudaGrid));
  cudaGrid.shiftInTime ();

  dim3 blocks (cudaGrid.getSize ().get1 () / 2, cudaGrid.getSize ().get2 () / 2, cudaGrid.getSize ().get3 () / 2);
  dim3 threads (2, 2, 2);
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    for (grid_coord j = bufSize.get2 (); j < cudaGrid.getSize ().get2 () - bufSize.get2 (); ++j)
    {
      for (grid_coord k = bufSize.get3 (); k < cudaGrid.getSize ().get3 () - bufSize.get3 (); ++k)
      {
        GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k,
                                                   overallSize.getType1 (),
                                                   overallSize.getType2 (),
                                                   overallSize.getType3 ());
        grid_coord index = cudaGrid.calculateIndexFromPosition (pos);

        GridCoordinate3D posCPU = GRID_COORDINATE_3D (i - bufSize.get1 (), j - bufSize.get2 (), k - bufSize.get3 (),
                                                      overallSize.getType1 (),
                                                      overallSize.getType2 (),
                                                      overallSize.getType3 ());
        grid_coord coord = cpuGrid.calculateIndexFromPosition (posCPU);

        ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 0) == FIELDVALUE (17, 1022) + FIELDVALUE (index * 23, index * 17));
        ALWAYS_ASSERT (*cpuGrid.getFieldValue (coord, 1) == FIELDVALUE (17, 1022));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (zero, 0) == cudaGrid2.getFieldValue (bufSize, 0));
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
    ASSERT (cudaGrid2.getFieldValueByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getFieldValueOrNullByAbsolutePos (overallSize / 2, 0) == cudaGrid2.getFieldValue (bufSize, 0));
    ASSERT (cudaGrid2.getComputationStart (one) == one);
    ASSERT (cudaGrid2.getComputationEnd (one) == overallSize / 2 + bufSize - one);
    ASSERT (cudaGrid2.getHasLeft ().get1 () == 1);
    ASSERT (cudaGrid2.getHasRight ().get1 () == 0);
  }
}
#endif /* MODE_DIM3 */

int main (int argc, char** argv)
{
  cudaCheckErrorCmd (cudaSetDevice(STOI (argv[1])));
  solverSettings.Initialize ();

  int gridSizeX = 32;
  int gridSizeY = 46;
  int gridSizeZ = 40;

#if defined (MODE_DIM1)
  testFunc1D (GridCoordinate1D (gridSizeX, CoordinateType::X),
              GridCoordinate1D (1, CoordinateType::X));
  testFunc1D (GridCoordinate1D (gridSizeY, CoordinateType::Y),
              GridCoordinate1D (1, CoordinateType::Y));
  testFunc1D (GridCoordinate1D (gridSizeZ, CoordinateType::Z),
              GridCoordinate1D (1, CoordinateType::Z));

  testFunc1D (GridCoordinate1D (gridSizeX, CoordinateType::X),
              GridCoordinate1D (0, CoordinateType::X));
  testFunc1D (GridCoordinate1D (gridSizeY, CoordinateType::Y),
              GridCoordinate1D (0, CoordinateType::Y));
  testFunc1D (GridCoordinate1D (gridSizeZ, CoordinateType::Z),
              GridCoordinate1D (0, CoordinateType::Z));
#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)
  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y),
              GridCoordinate2D (1, 1, CoordinateType::X, CoordinateType::Y));
  testFunc2D (GridCoordinate2D (gridSizeY, gridSizeZ, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate2D (1, 1, CoordinateType::Y, CoordinateType::Z));
  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeZ, CoordinateType::X, CoordinateType::Z),
              GridCoordinate2D (1, 1, CoordinateType::X, CoordinateType::Z));

  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y),
              GridCoordinate2D (0, 0, CoordinateType::X, CoordinateType::Y));
  testFunc2D (GridCoordinate2D (gridSizeY, gridSizeZ, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate2D (0, 0, CoordinateType::Y, CoordinateType::Z));
  testFunc2D (GridCoordinate2D (gridSizeX, gridSizeZ, CoordinateType::X, CoordinateType::Z),
              GridCoordinate2D (0, 0, CoordinateType::X, CoordinateType::Z));
#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)
  testFunc3D (GridCoordinate3D (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate3D (1, 1, 1, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
  testFunc3D (GridCoordinate3D (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
              GridCoordinate3D (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
#endif /* MODE_DIM3 */

  solverSettings.Uninitialize ();

  return 0;
} /* main */

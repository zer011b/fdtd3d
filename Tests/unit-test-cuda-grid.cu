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

  /*
   * CudaGrid with capacity for all cpu grid
   */
  CudaGrid<GridCoordinate1D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 4, 1, 1);
  dim3 threads (4, 1, 1);

  CudaGrid<GridCoordinate1D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate1D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    GridCoordinate1D pos (i - bufSize.get1 ()
#ifdef DEBUG_INFO
                          , overallSize.getType1 ()
#endif /* DEBUG_INFO */
                         );
    FieldPointValue *val = cpuGrid.getFieldPointValue (pos);
    grid_coord index = pos.calculateTotalCoord ();

    ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
    ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
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

  /*
   * CudaGrid with capacity for all cpu grid
   */
  CudaGrid<GridCoordinate2D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 4, cudaGrid.getSize ().get2 () / 4, 1);
  dim3 threads (4, 4, 1);

  CudaGrid<GridCoordinate2D> *d_cudaGrid;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaGrid, sizeof (CudaGrid<GridCoordinate2D>)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaGrid, &cudaGrid, sizeof(CudaGrid<GridCoordinate2D>), cudaMemcpyHostToDevice));
  cudaCheckExitStatus (cudaCalculate <<< blocks, threads >>> (exitStatusCuda, d_cudaGrid));

  cudaGrid.copyToCPU ();

  for (grid_coord i = bufSize.get1 (); i < cudaGrid.getSize ().get1 () - bufSize.get1 (); ++i)
  {
    for (grid_coord j = bufSize.get2 (); j < cudaGrid.getSize ().get2 () - bufSize.get2 (); ++j)
    {
      GridCoordinate2D pos (i - bufSize.get1 (), j - bufSize.get2 ()
#ifdef DEBUG_INFO
                            , overallSize.getType1 ()
                            , overallSize.getType2 ()
#endif /* DEBUG_INFO */
                           );
      FieldPointValue *val = cpuGrid.getFieldPointValue (pos);
      grid_coord index = pos.calculateTotalCoord ();

      ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
      ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
    }
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

  /*
   * CudaGrid with capacity for all cpu grid
   */
  CudaGrid<GridCoordinate3D> cudaGrid (overallSize, bufSize, &cpuGrid);
  cudaGrid.copyFromCPU (zero, overallSize);

  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (cudaGrid.getSize ().get1 () / 4, cudaGrid.getSize ().get2 () / 4, cudaGrid.getSize ().get3 () / 4);
  dim3 threads (4, 4, 4);

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
        GridCoordinate3D pos (i - bufSize.get1 (), j - bufSize.get2 (), k - bufSize.get3 ()
#ifdef DEBUG_INFO
                              , overallSize.getType1 ()
                              , overallSize.getType2 ()
                              , overallSize.getType3 ()
#endif /* DEBUG_INFO */
                             );
        FieldPointValue *val = cpuGrid.getFieldPointValue (pos);
        grid_coord index = pos.calculateTotalCoord ();

        ALWAYS_ASSERT (val->getCurValue () == FIELDVALUE (index * 23, index * 17));
        ALWAYS_ASSERT (val->getPrevValue () == FIELDVALUE (17, 1022));
      }
    }
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

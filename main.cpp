#include <iostream>

#include <ctime>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#include <mpi.h>
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#ifdef CUDA_ENABLED
#include "CudaInterface.h"
#endif

#ifdef CXX11_ENABLED
#else
#include "cstdlib"
#endif

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "SchemeTMz.h"
#include "Scheme3D.h"

int main (int argc, char** argv)
{
  int totalTimeSteps = 100;
  int gridSize = 100;
  int bufSize = 10;

  int dimension;
  bool is_parallel_grid;

  if (argc != 4)
  {
    return 1;
  }
  else
  {
#ifdef CXX11_ENABLED
    totalTimeSteps = std::stoi (argv[1]);
    gridSize = std::stoi (argv[2]);
    bufSize = std::stoi (argv[3]);
#else
    totalTimeSteps = atoi (argv[1]);
    gridSize = atoi (argv[2]);
    bufSize = atoi (argv[3]);
#endif
  }

  const clock_t begin_time = clock();

#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#if PRINT_MESSAGE
  printf ("Start process %d of %d\n", rank, numProcs);
#endif /* PRINT_MESSAGE */

  is_parallel_grid = true;
#else /* PARALLEL_GRID */
  is_parallel_grid = false;
#endif /* !PARALLEL_GRID */

#ifdef CUDA_ENABLED
  cudaInit (0);
#endif

#ifdef GRID_2D
  dimension = 2;
#endif
#ifdef GRID_3D
  dimension = 3;
#endif

#if defined (PARALLEL_GRID)
  ParallelGridCoordinate overallSize (gridSize);
  ParallelGridCoordinate bufferLeft (bufSize);
  ParallelGridCoordinate bufferRight (bufSize);

#ifdef GRID_2D
  SchemeTMz scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps);
#endif
#ifdef GRID_3D
  Scheme3D scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps);
#endif
#else
#ifdef GRID_2D
  GridCoordinate2D overallSize (gridSize);

  SchemeTMz scheme (overallSize, totalTimeSteps);
#endif
#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSize);

  Scheme3D scheme (overallSize, totalTimeSteps);
#endif
#endif

  scheme.initScheme (0.0000003, 20);

#if defined (PARALLEL_GRID)
  scheme.initProcess (rank);
#endif

  scheme.initGrids ();

  scheme.performSteps ();

#if defined (PARALLEL_GRID)
#if PRINT_MESSAGE
  printf ("Main process %d.\n", rank);
#endif

  MPI_Finalize();
#endif

#if defined (PARALLEL_GRID)
  if (rank == 0)
  {
#endif /* PARALLEL_GRID */

    const clock_t end_time = clock();

    printf ("Dimension: %d\n", dimension);
    printf ("Grid size: %d\n", gridSize);
    printf ("Number of time steps: %d\n", totalTimeSteps);

    printf ("\n");

#ifdef FLOAT_VALUES
    printf ("Value type: float\n");
#endif
#ifdef DOUBLE_VALUES
    printf ("Value type: double\n");
#endif
#ifdef LONG_DOUBLE_VALUES
    printf ("Value type: long double\n");
#endif

#ifdef TWO_TIME_STEPS
    printf ("Number of time steps: 2\n");
#endif
#ifdef ONE_TIME_STEP
    printf ("Number of time steps: 1\n");
#endif

    printf ("\n-------- Details --------\n");
    printf ("Parallel grid: %d\n", is_parallel_grid);

#if defined (PARALLEL_GRID)
    printf ("Number of processes: %d\n", numProcs);

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
    printf ("Parallel grid scheme: X\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
    printf ("Parallel grid scheme: Y\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
    printf ("Parallel grid scheme: Z\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
    printf ("Parallel grid scheme: XY\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
    printf ("Parallel grid scheme: YZ\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
    printf ("Parallel grid scheme: XZ\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
    printf ("Parallel grid scheme: XYZ\n");
#endif

    printf ("Buffer size: %d\n", bufSize);
#endif

    printf ("\n-------- Execution Time --------\n");

    FieldValue execution_time = (FieldValue) (((FieldValue) (end_time - begin_time)) /  CLOCKS_PER_SEC);

    printf ("Execution time (by clock()): %f seconds.\n", execution_time);

#if defined (PARALLEL_GRID)
  }
#endif /* PARALLEL_GRID */

  return 0;
}

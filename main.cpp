#include <iostream>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#include <mpi.h>
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "SchemeTMz.h"

int main (int argc, char** argv)
{
  int totalTimeSteps = 100;

  int gridSize = 256;

#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#if PRINT_MESSAGE
  printf ("Start process %d of %d\n", rank, numProcs);
#endif /* PRINT_MESSAGE */
#endif /* PARALLEL_GRID */

#if defined (PARALLEL_GRID)
  ParallelGridCoordinate overallSize (gridSize);
  ParallelGridCoordinate bufferLeft (30);
  ParallelGridCoordinate bufferRight (30);

  SchemeTMz scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps);
#else
  GridCoordinate2D overallSize (gridSize);

  SchemeTMz scheme (overallSize, totalTimeSteps);
#endif

  scheme.initScheme (0.000003, 20);

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

  return 0;
}

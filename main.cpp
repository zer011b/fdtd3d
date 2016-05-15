#include <iostream>

#include <mpi.h>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "SchemeTEz.h"

int main (int argc, char** argv)
{
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
  ParallelGridCoordinate overallSize (100);
  //GridCoordinate size (100, 100);
  ParallelGridCoordinate bufferLeft (10);
  ParallelGridCoordinate bufferRight (10);

  SchemeTEz scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, 100);
#else
  GridCoordinate2D overallSize (100);

  SchemeTEz scheme (overallSize, 120);
#endif

  scheme.initScheme (0.000003, 20);

#if defined (PARALLEL_GRID)
  scheme.initProcess (rank);
#endif

  scheme.initGrids ();

  scheme.performStep ();

#if defined (PARALLEL_GRID)
#if PRINT_MESSAGE
  printf ("Main process %d.\n", rank);
#endif

  MPI_Finalize();
#endif

  return 0;
}

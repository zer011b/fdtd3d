#include "Grid.h"

#ifdef GRID_3D


#ifdef PARALLEL_BUFFER_DIMENSION_1D
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D */

#ifdef PARALLEL_BUFFER_DIMENSION_2D
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_2D */

#ifdef PARALLEL_BUFFER_DIMENSION_3D
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D)
void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D ||
          PARALLEL_BUFFER_DIMENSION_2D ||
          PARALLEL_BUFFER_DIMENSION_3D */


#endif /* GRID_3D */

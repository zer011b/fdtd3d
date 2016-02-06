#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_3D


#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X ||
          PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#endif /* GRID_3D */

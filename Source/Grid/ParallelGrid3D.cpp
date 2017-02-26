#include "ParallelGrid.h"

#include <cmath>

#ifdef PARALLEL_GRID

#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

/**
 * Initialize 3D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
  nodeGridSizeZ = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = totalProcCount;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d.\n",
          processId,
          nodeGridSizeX,
          nodeGridSizeY,
          nodeGridSizeZ);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y ||
          PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

/**
 * Initialize 3D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  NodeGridInitInner (desiredProportion.getX (), nodeGridSizeTmp1, nodeGridSizeTmp2, left);

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = 1;

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeYZ = nodeGridSizeY * nodeGridSizeZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeXZ = nodeGridSizeX * nodeGridSizeZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n",
          processId,
          nodeGridSizeX,
          nodeGridSizeY,
          nodeGridSizeZ,
          left);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Initialize 3D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
  if (totalProcCount < 8)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  int nodeGridSizeTmp3;
  NodeGridInitInner (desiredProportion.getX (), desiredProportion.getY (), nodeGridSizeTmp1, nodeGridSizeTmp2, nodeGridSizeTmp3, left);

  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = nodeGridSizeTmp3;

  nodeGridSizeXYZ = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n",
          processId,
          nodeGridSizeX,
          nodeGridSizeY,
          nodeGridSizeZ,
          left);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */

#endif /* GRID_3D */

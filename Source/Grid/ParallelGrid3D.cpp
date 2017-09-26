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
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< size of grid */
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  if (!doUseManualTopology)
  {
    nodeGridSizeX = totalProcCount;
  }
  else
  {
    nodeGridSizeX = topologySize.getX ();
  }
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  if (!doUseManualTopology)
  {
    nodeGridSizeY = totalProcCount;
  }
  else
  {
    nodeGridSizeY = topologySize.getY ();
  }
  nodeGridSizeX = 1;
  nodeGridSizeZ = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  if (!doUseManualTopology)
  {
    nodeGridSizeZ = totalProcCount;
  }
  else
  {
    nodeGridSizeZ = topologySize.getZ ();
  }
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "manual" : "optimal",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);
  }
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
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< desired relation values */
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (!doUseManualTopology)
  {
    initOptimal (size.getX (), size.getY (), nodeGridSizeX, nodeGridSizeY);
  }
  else
  {
    nodeGridSizeX = topologySize.getX ();
    nodeGridSizeY = topologySize.getY ();
  }
  nodeGridSizeZ = 1;

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (!doUseManualTopology)
  {
    initOptimal (size.getY (), size.getZ (), nodeGridSizeY, nodeGridSizeZ);
  }
  else
  {
    nodeGridSizeY = topologySize.getY ();
    nodeGridSizeZ = topologySize.getZ ();
  }
  nodeGridSizeX = 1;

  nodeGridSizeYZ = nodeGridSizeY * nodeGridSizeZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (!doUseManualTopology)
  {
    initOptimal (size.getX (), size.getZ (), nodeGridSizeX, nodeGridSizeZ);
  }
  else
  {
    nodeGridSizeX = topologySize.getX ();
    nodeGridSizeZ = topologySize.getZ ();
  }
  nodeGridSizeY = 1;

  nodeGridSizeXZ = nodeGridSizeX * nodeGridSizeZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "manual" : "optimal",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);
  }
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Initialize 3D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< size of grid */
{
  if (totalProcCount < 8)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  if (!doUseManualTopology)
  {
    initOptimal (size.getX (), size.getY (), size.getZ (), nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ);
  }
  else
  {
    nodeGridSizeX = topologySize.getX ();
    nodeGridSizeY = topologySize.getY ();
    nodeGridSizeZ = topologySize.getZ ();
  }

  nodeGridSizeXYZ = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "manual" : "optimal",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);
  }
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */

#endif /* GRID_3D */

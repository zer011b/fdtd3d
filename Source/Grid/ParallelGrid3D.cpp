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
  int nodeGridSizeXOptimal;
  int nodeGridSizeYOptimal;
  int nodeGridSizeZOptimal;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeXOptimal = totalProcCount;
  nodeGridSizeYOptimal = 1;
  nodeGridSizeZOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
  }
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;

  if (nodeGridSizeX <= 1)
  {
    ASSERT_MESSAGE ("3D-X virtual topology could be used only with number of processes > 1 by Ox axis. "
                    "Use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeXOptimal = 1;
  nodeGridSizeYOptimal = totalProcCount;
  nodeGridSizeZOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeY = nodeGridSizeYOptimal;
  }
  else
  {
    nodeGridSizeY = topologySize.get2 ();
  }
  nodeGridSizeX = 1;
  nodeGridSizeZ = 1;

  if (nodeGridSizeY <= 1)
  {
    ASSERT_MESSAGE ("3D-Y virtual topology could be used only with number of processes > 1 by Oy axis. "
                    "Use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeXOptimal = 1;
  nodeGridSizeYOptimal = 1;
  nodeGridSizeZOptimal = totalProcCount;

  if (!doUseManualTopology)
  {
    nodeGridSizeZ = nodeGridSizeZOptimal;
  }
  else
  {
    nodeGridSizeZ = topologySize.get3 ();
  }
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;

  if (nodeGridSizeZ <= 1)
  {
    ASSERT_MESSAGE ("3D-Z virtual topology could be used only with number of processes > 1 by Oz axis. "
                    "Use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%dx%dx%d). Consider using OPTIMAL virtual topology (%dx%dx%d). \n",
              nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ,
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%dx%dx%d).\n",
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }

    printf ("OPTIMAL virtual topology has some requirements in order for it to be optimal:\n"
            "  - all computational nodes have the same performance\n"
            "  - all neighboring computational nodes have the same share time\n"
            "  - virtual topology matches physical topology\n"
            "In other words, OPTIMAL virtual topology is optimal only for homogeneous architectures.\n"
            "Make sure all these requirements are met, otherwise, OPTIMAL virtual topology could be non-optimal.\n"
            "===================================================================================================\n");
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

  int nodeGridSizeXOptimal;
  int nodeGridSizeYOptimal;
  int nodeGridSizeZOptimal;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  initOptimal (size.get1 (), size.get2 (), nodeGridSizeXOptimal, nodeGridSizeYOptimal);
  nodeGridSizeZOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
    nodeGridSizeY = nodeGridSizeYOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
    nodeGridSizeY = topologySize.get2 ();
  }
  nodeGridSizeZ = 1;

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  if (nodeGridSizeX <= 1 || nodeGridSizeY <= 1)
  {
    ASSERT_MESSAGE ("3D-XY virtual topology could be used only with number of processes > 1 by Ox and Oy axis. "
                    "Recompile with `-DPARALLEL_BUFFER_DIMENSION=x`, or `-DPARALLEL_BUFFER_DIMENSION=y`, or "
                    "use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  initOptimal (size.get2 (), size.get3 (), nodeGridSizeYOptimal, nodeGridSizeZOptimal);
  nodeGridSizeXOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeY = nodeGridSizeYOptimal;
    nodeGridSizeZ = nodeGridSizeZOptimal;
  }
  else
  {
    nodeGridSizeY = topologySize.get2 ();
    nodeGridSizeZ = topologySize.get3 ();
  }
  nodeGridSizeX = 1;

  nodeGridSizeYZ = nodeGridSizeY * nodeGridSizeZ;

  if (nodeGridSizeY <= 1 || nodeGridSizeZ <= 1)
  {
    ASSERT_MESSAGE ("3D-YZ virtual topology could be used only with number of processes > 1 by Oy and Oz axis. "
                    "Recompile with `-DPARALLEL_BUFFER_DIMENSION=y`, or `-DPARALLEL_BUFFER_DIMENSION=z`, or "
                    "use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  initOptimal (size.get1 (), size.get3 (), nodeGridSizeXOptimal, nodeGridSizeZOptimal);
  nodeGridSizeYOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
    nodeGridSizeZ = nodeGridSizeZOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
    nodeGridSizeZ = topologySize.get3 ();
  }
  nodeGridSizeY = 1;

  nodeGridSizeXZ = nodeGridSizeX * nodeGridSizeZ;

  if (nodeGridSizeX <= 1 || nodeGridSizeZ <= 1)
  {
    ASSERT_MESSAGE ("3D-XZ virtual topology could be used only with number of processes > 1 by Ox and Oz axis. "
                    "Recompile with `-DPARALLEL_BUFFER_DIMENSION=x`, or `-DPARALLEL_BUFFER_DIMENSION=z`, or "
                    "use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%dx%dx%d). Consider using OPTIMAL virtual topology (%dx%dx%d). \n",
              nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ,
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%dx%dx%d).\n",
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }

    printf ("OPTIMAL virtual topology has some requirements in order for it to be optimal:\n"
            "  - all computational nodes have the same performance\n"
            "  - all neighboring computational nodes have the same share time\n"
            "  - virtual topology matches physical topology\n"
            "In other words, OPTIMAL virtual topology is optimal only for homogeneous architectures.\n"
            "Make sure all these requirements are met, otherwise, OPTIMAL virtual topology could be non-optimal.\n"
            "===================================================================================================\n");
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

  int nodeGridSizeXOptimal;
  int nodeGridSizeYOptimal;
  int nodeGridSizeZOptimal;
  initOptimal (size.get1 (), size.get2 (), size.get3 (), nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
    nodeGridSizeY = nodeGridSizeYOptimal;
    nodeGridSizeZ = nodeGridSizeZOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
    nodeGridSizeY = topologySize.get2 ();
    nodeGridSizeZ = topologySize.get3 ();
  }

  nodeGridSizeXYZ = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  if (nodeGridSizeX <= 1 || nodeGridSizeY <= 1 || nodeGridSizeZ <= 1)
  {
    ASSERT_MESSAGE ("3D-XYZ virtual topology could be used only with number of processes > 1 by Ox and Oy axis. "
                    "Recompile with `-DPARALLEL_BUFFER_DIMENSION=x`, or `-DPARALLEL_BUFFER_DIMENSION=y`, or "
                    "`-DPARALLEL_BUFFER_DIMENSION=z`, or `-DPARALLEL_BUFFER_DIMENSION=xy`, or "
                    "`-DPARALLEL_BUFFER_DIMENSION=yz`, or `-DPARALLEL_BUFFER_DIMENSION=xz`, or "
                    "use without parallel grid");
  }

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%dx%d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX,
            nodeGridSizeY,
            nodeGridSizeZ);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%dx%dx%d). Consider using OPTIMAL virtual topology (%dx%dx%d). \n",
              nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ,
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%dx%dx%d).\n",
              nodeGridSizeXOptimal, nodeGridSizeYOptimal, nodeGridSizeZOptimal);
    }

    printf ("OPTIMAL virtual topology has some requirements in order for it to be optimal:\n"
            "  - all computational nodes have the same performance\n"
            "  - all neighboring computational nodes have the same share time\n"
            "  - virtual topology matches physical topology\n"
            "In other words, OPTIMAL virtual topology is optimal only for homogeneous architectures.\n"
            "Make sure all these requirements are met, otherwise, OPTIMAL virtual topology could be non-optimal.\n"
            "===================================================================================================\n");
  }
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */

#endif /* GRID_3D */

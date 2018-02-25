#include "ParallelGrid.h"

#ifdef GRID_1D

#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X

/**
 * Initialize 1D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< size of grid */
{
  int nodeGridSizeXOptimal = totalProcCount;

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
  }

  if (nodeGridSizeX <= 1)
  {
    ASSERT_MESSAGE ("1D-X virtual topology could be used only with number of processes > 1 by Ox axis. "
                    "Use without parallel grid");
  }

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%d). Consider using OPTIMAL virtual topology (%d). \n",
              nodeGridSizeX, nodeGridSizeXOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%d).\n",
              nodeGridSizeXOptimal);
    }

    printf ("OPTIMAL virtual topology has some requirements in order for it to be optimal:\n"
            "  - all computational nodes have the same performance\n"
            "  - all neighbouring computational nodes have the same share time\n"
            "  - virtual topology matches physical topology\n"
            "In other words, OPTIMAL virtual topology is optimal only for homogeneous architectures.\n"
            "Make sure all these requirements are met, otherwise, OPTIMAL virtual topology could be non-optimal.\n"
            "===================================================================================================\n");
  }
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#endif /* PARALLEL_GRID */

#endif /* GRID_1D */

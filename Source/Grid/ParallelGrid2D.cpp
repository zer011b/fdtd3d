/*
 * Copyright (C) 2016 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "ParallelGrid.h"

#ifdef GRID_2D

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

/**
 * Initialize 2D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< size of grid */
{
  int nodeGridSizeXOptimal;
  int nodeGridSizeYOptimal;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeXOptimal = totalProcCount;
  nodeGridSizeYOptimal = 1;

  if (!doUseManualTopology)
  {
    nodeGridSizeX = nodeGridSizeXOptimal;
  }
  else
  {
    nodeGridSizeX = topologySize.get1 ();
  }
  nodeGridSizeY = 1;

  if (nodeGridSizeX <= 1)
  {
    ASSERT_MESSAGE ("2D-X virtual topology could be used only with number of processes > 1 by Ox axis. "
                    "Use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeXOptimal = 1;
  nodeGridSizeYOptimal = totalProcCount;

  if (!doUseManualTopology)
  {
    nodeGridSizeY = nodeGridSizeYOptimal;
  }
  else
  {
    nodeGridSizeY = topologySize.get2 ();
  }
  nodeGridSizeX = 1;

  if (nodeGridSizeY <= 1)
  {
    ASSERT_MESSAGE ("2D-Y virtual topology could be used only with number of processes > 1 by Oy axis. "
                    "Use without parallel grid");
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX,
            nodeGridSizeY);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%dx%d). Consider using OPTIMAL virtual topology (%dx%d). \n",
              nodeGridSizeX, nodeGridSizeY, nodeGridSizeXOptimal, nodeGridSizeYOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%dx%d).\n",
              nodeGridSizeXOptimal, nodeGridSizeYOptimal);
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

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY

/**
 * Initialize 2D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinate size) /**< size of grid */
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  int nodeGridSizeXOptimal;
  int nodeGridSizeYOptimal;
  initOptimal (size.get1 (), size.get2 (), nodeGridSizeXOptimal, nodeGridSizeYOptimal);

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

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  if (nodeGridSizeX <= 1 || nodeGridSizeY <= 1)
  {
    ASSERT_MESSAGE ("2D-XY virtual topology could be used only with number of processes > 1 by Ox and Oy axis. "
                    "Recompile with `-DPARALLEL_BUFFER_DIMENSION=x`, or `-DPARALLEL_BUFFER_DIMENSION=y`, or "
                    "use without parallel grid");
  }

  if (getProcessId () == 0)
  {
    printf ("Nodes' grid (%s): %dx%d.\n",
            doUseManualTopology ? "MANUAL" : "OPTIMAL",
            nodeGridSizeX,
            nodeGridSizeY);

    printf ("===================================================================================================\n");

    if (doUseManualTopology)
    {
      printf ("NOTE: you use MANUAL virtual topology (%dx%d). Consider using OPTIMAL virtual topology (%dx%d). \n",
              nodeGridSizeX, nodeGridSizeY, nodeGridSizeXOptimal, nodeGridSizeYOptimal);
    }
    else
    {
      printf ("NOTE: you use OPTIMAL virtual topology (%dx%d).\n",
              nodeGridSizeXOptimal, nodeGridSizeYOptimal);
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

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#endif /* PARALLEL_GRID */

#endif /* GRID_2D */

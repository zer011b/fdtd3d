/*
 * Copyright (C) 2018 Gleb Balykov
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

#ifdef PARALLEL_GRID

#ifdef DYNAMIC_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Estimate share time across border between pidX1 and pidX2. This is just estimation, which is used to determine the
 * border with the highest share overhead.
 *
 * @return share time for specified buffer size
 *         0, if share time can't be estimated
 */
template <SchemeType_t Type, LayoutType layout_type>
DOUBLE
ParallelYeeGridLayout<Type, layout_type>::estimateTimeAcrossAxisX (int pidX1, /**< x coordinate of first process plane */
                                                                   int pidX2) const /**< x coordinate of second process plane */
{
  bool enabled1 = false;
  bool enabled2 = false;

#if defined (GRID_2D) || defined (GRID_3D)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
#endif
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_1D
    int pid1 = parallelGridCore->getNodeGrid (pidX1);
    int pid2 = parallelGridCore->getNodeGrid (pidX2);
#endif
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      enabled1 = true;
    }

    if (parallelGridCore->getNodeState ()[pid2] == 1)
    {
      enabled2 = true;
    }
  }

  if (!enabled1 || !enabled2)
  {
    /*
     * One of two processes is disabled, so, no latency for them
     */
    return 0.0;
  }

  int tmpCounter = 0;

  DOUBLE latencyX = DOUBLE (0);
  DOUBLE bandwidthX = DOUBLE (0);

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyX_diag_up = DOUBLE (0);
  DOUBLE latencyX_diag_down = DOUBLE (0);
  DOUBLE bandwidthX_diag_up = DOUBLE (0);
  DOUBLE bandwidthX_diag_down = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyX_diag_back = DOUBLE (0);
  DOUBLE latencyX_diag_front = DOUBLE (0);
  DOUBLE bandwidthX_diag_back = DOUBLE (0);
  DOUBLE bandwidthX_diag_front = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyX_diag_up_back = DOUBLE (0);
  DOUBLE latencyX_diag_up_front = DOUBLE (0);
  DOUBLE latencyX_diag_down_back = DOUBLE (0);
  DOUBLE latencyX_diag_down_front = DOUBLE (0);

  DOUBLE bandwidthX_diag_up_back = DOUBLE (0);
  DOUBLE bandwidthX_diag_up_front = DOUBLE (0);
  DOUBLE bandwidthX_diag_down_back = DOUBLE (0);
  DOUBLE bandwidthX_diag_down_front = DOUBLE (0);
#endif

  tmpCounter = 0;

#if defined (GRID_2D) || defined (GRID_3D)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
#endif
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_1D
    int pid1 = parallelGridCore->getNodeGrid (pidX1);
    int pid2 = parallelGridCore->getNodeGrid (pidX2);
#endif
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j, k);
#endif

    ASSERT (parallelGridCore->getNodeState ()[pid1] == parallelGridCore->getNodeState ()[pid2]);
    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      latencyX += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX /= tmpCounter;
    bandwidthX /= tmpCounter;

    if (latencyX < 0 || bandwidthX <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX = 0;
    bandwidthX = 0;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j + 1);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j + 1, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_up += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_up += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_up /= tmpCounter;
    bandwidthX_diag_up /= tmpCounter;

    if (latencyX_diag_up < 0 || bandwidthX_diag_up <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_up = 0;
    bandwidthX_diag_up = 0;
  }

  tmpCounter = 0;

  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j - 1);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j - 1, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_down += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_down += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_down /= tmpCounter;
    bandwidthX_diag_down /= tmpCounter;

    if (latencyX_diag_down < 0 || bandwidthX_diag_down <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_down = 0;
    bandwidthX_diag_down = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_front /= tmpCounter;
    bandwidthX_diag_front /= tmpCounter;

    if (latencyX_diag_front < 0 || bandwidthX_diag_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_front = 0;
    bandwidthX_diag_front = 0;
  }

  tmpCounter = 0;

  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_back /= tmpCounter;
    bandwidthX_diag_back /= tmpCounter;

    if (latencyX_diag_back < 0 || bandwidthX_diag_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_back = 0;
    bandwidthX_diag_back = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j + 1, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_up_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_up_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_up_front /= tmpCounter;
    bandwidthX_diag_up_front /= tmpCounter;

    if (latencyX_diag_up_front < 0 || bandwidthX_diag_up_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_up_front = 0;
    bandwidthX_diag_up_front = 0;
  }

  tmpCounter = 0;

  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j + 1, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_up_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_up_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_up_back /= tmpCounter;
    bandwidthX_diag_up_back /= tmpCounter;

    if (latencyX_diag_up_back < 0 || bandwidthX_diag_up_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_up_back = 0;
    bandwidthX_diag_up_back = 0;
  }

  tmpCounter = 0;

  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j - 1, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_down_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_down_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_down_front /= tmpCounter;
    bandwidthX_diag_down_front /= tmpCounter;

    if (latencyX_diag_down_front < 0 || bandwidthX_diag_down_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_down_front = 0;
    bandwidthX_diag_down_front = 0;
  }

  tmpCounter = 0;

  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (pidX1, j, k);
    int pid2 = parallelGridCore->getNodeGrid (pidX2, j - 1, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyX_diag_down_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthX_diag_down_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyX_diag_down_back /= tmpCounter;
    bandwidthX_diag_down_back /= tmpCounter;

    if (latencyX_diag_down_back < 0 || bandwidthX_diag_down_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyX_diag_down_back = 0;
    bandwidthX_diag_down_back = 0;
  }
#endif

  DOUBLE val = DOUBLE (0);

  ParallelGridCoordinate size = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size;

#ifdef GRID_1D
  DOUBLE sizeX = DOUBLE (1);
#endif
#ifdef GRID_2D
  DOUBLE sizeX = DOUBLE (size.get2 ());
#endif
#ifdef GRID_3D
  DOUBLE sizeX = DOUBLE (size.get2 () * size.get3 ());
#endif

  if (latencyX >= 0 && bandwidthX > 0)
  {
    val += latencyX + sizeX / bandwidthX;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
#ifdef GRID_2D
  DOUBLE sizeX_diag_down = DOUBLE (1);
  DOUBLE sizeX_diag_up = DOUBLE (1);
#endif
#ifdef GRID_3D
  DOUBLE sizeX_diag_down = DOUBLE (size.get3 ());
  DOUBLE sizeX_diag_up = DOUBLE (size.get3 ());
#endif

  if (latencyX_diag_up >= 0 && latencyX_diag_down >= 0 && bandwidthX_diag_up > 0 && bandwidthX_diag_down > 0)
  {
    val += latencyX_diag_down + sizeX_diag_down / bandwidthX_diag_down
           + latencyX_diag_up + sizeX_diag_up / bandwidthX_diag_up;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE sizeX_diag_back = DOUBLE (size.get2 ());
  DOUBLE sizeX_diag_front = DOUBLE (size.get2 ());

  if (latencyX_diag_back >= 0 && latencyX_diag_front >= 0 && bandwidthX_diag_back > 0 && bandwidthX_diag_front > 0)
  {
    val += latencyX_diag_back + sizeX_diag_back / bandwidthX_diag_back
           + latencyX_diag_front + sizeX_diag_front / bandwidthX_diag_front;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (latencyX_diag_up_back >= 0 && latencyX_diag_up_front >= 0 && latencyX_diag_down_back >= 0 && latencyX_diag_down_front >= 0
      && bandwidthX_diag_up_back > 0 && bandwidthX_diag_up_front > 0 && bandwidthX_diag_down_back > 0 && bandwidthX_diag_down_front > 0)
  {
    val += latencyX_diag_down_back + 1.0 / bandwidthX_diag_down_back
           + latencyX_diag_down_front + 1.0 / bandwidthX_diag_down_front
           + latencyX_diag_up_back + 1.0 / bandwidthX_diag_up_back
           + latencyX_diag_up_front + 1.0 / bandwidthX_diag_up_front;
  }
#endif

  return val;
}

template <SchemeType_t Type, LayoutType layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::estimateBorderX (std::vector<NodeBorder_t> &borders,
                                                           const std::vector<grid_coord> &spreadX) const
{
  int i = 0;

  while (i < parallelGridCore->getNodeGridSizeX () - 1)
  {
    while (i < parallelGridCore->getNodeGridSizeX ()
           && spreadX[i] == 0)
    {
      ++i;
    }

    int j = i + 1;
    while (j < parallelGridCore->getNodeGridSizeX ()
           && spreadX[j] == 0)
    {
      ++j;
    }

    /*
     * Checking border between i and j
     */
    DOUBLE timeVal = estimateTimeAcrossAxisX (i, j);
    if (timeVal == DOUBLE (0))
    {
      /*
       * No time estimation is available
       */
      continue;
    }
    borders.push_back (NodeBorder_t (i, j, OX, timeVal));
  }
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Estimate share time across border between pidX1 and pidX2. This is just estimation, which is used to determine the
 * border with the highest share overhead.
 *
 * @return share time for specified buffer size
 *         0, if share time can't be estimated
 */
template <SchemeType_t Type, LayoutType layout_type>
DOUBLE
ParallelYeeGridLayout<Type, layout_type>::estimateTimeAcrossAxisY (int pidY1, /**< y coordinate of first process plane */
                                                                   int pidY2) const /**< y coordinate of second process plane */
{
  bool enabled1 = false;
  bool enabled2 = false;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      enabled1 = true;
    }

    if (parallelGridCore->getNodeState ()[pid2] == 1)
    {
      enabled2 = true;
    }
  }

  if (!enabled1 || !enabled2)
  {
    /*
     * One of two processes is disabled, so, no latency for them
     */
    return 0.0;
  }

  int tmpCounter = 0;

  DOUBLE latencyY = DOUBLE (0);
  DOUBLE bandwidthY = DOUBLE (0);

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyY_diag_right = DOUBLE (0);
  DOUBLE latencyY_diag_left = DOUBLE (0);
  DOUBLE bandwidthY_diag_right = DOUBLE (0);
  DOUBLE bandwidthY_diag_left = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyY_diag_back = DOUBLE (0);
  DOUBLE latencyY_diag_front = DOUBLE (0);
  DOUBLE bandwidthY_diag_back = DOUBLE (0);
  DOUBLE bandwidthY_diag_front = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyY_diag_left_back = DOUBLE (0);
  DOUBLE latencyY_diag_left_front = DOUBLE (0);
  DOUBLE latencyY_diag_right_back = DOUBLE (0);
  DOUBLE latencyY_diag_right_front = DOUBLE (0);

  DOUBLE bandwidthY_diag_left_back = DOUBLE (0);
  DOUBLE bandwidthY_diag_left_front = DOUBLE (0);
  DOUBLE bandwidthY_diag_right_back = DOUBLE (0);
  DOUBLE bandwidthY_diag_right_front = DOUBLE (0);
#endif

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2, k);
#endif

    ASSERT (parallelGridCore->getNodeState ()[pid1] == parallelGridCore->getNodeState ()[pid2]);
    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      latencyY += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY /= tmpCounter;
    bandwidthY /= tmpCounter;

    if (latencyY < 0 || bandwidthY <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY = 0;
    bandwidthY = 0;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, pidY2);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, pidY2, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_right += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_right += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_right /= tmpCounter;
    bandwidthY_diag_right /= tmpCounter;

    if (latencyY_diag_right < 0 || bandwidthY_diag_right <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_right = 0;
    bandwidthY_diag_right = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
#ifdef GRID_2D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, pidY2);
#endif
#ifdef GRID_3D
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, pidY2, k);
#endif

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_left += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_left += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_left /= tmpCounter;
    bandwidthY_diag_left /= tmpCounter;

    if (latencyY_diag_left < 0 || bandwidthY_diag_left <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_left = 0;
    bandwidthY_diag_left = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_front /= tmpCounter;
    bandwidthY_diag_front /= tmpCounter;

    if (latencyY_diag_front < 0 || bandwidthY_diag_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_front = 0;
    bandwidthY_diag_front = 0;
  }

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i, pidY2, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_back /= tmpCounter;
    bandwidthY_diag_back /= tmpCounter;

    if (latencyY_diag_back < 0 || bandwidthY_diag_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_back = 0;
    bandwidthY_diag_back = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, pidY2, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_right_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_right_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_right_front /= tmpCounter;
    bandwidthY_diag_right_front /= tmpCounter;

    if (latencyY_diag_right_front < 0 || bandwidthY_diag_right_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_right_front = 0;
    bandwidthY_diag_right_front = 0;
  }

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, pidY2, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_right_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_right_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_right_back /= tmpCounter;
    bandwidthY_diag_right_back /= tmpCounter;

    if (latencyY_diag_right_back < 0 || bandwidthY_diag_right_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_right_back = 0;
    bandwidthY_diag_right_back = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, pidY2, k + 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_left_front += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_left_front += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_left_front /= tmpCounter;
    bandwidthY_diag_left_front /= tmpCounter;

    if (latencyY_diag_left_front < 0 || bandwidthY_diag_left_front <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_left_front = 0;
    bandwidthY_diag_left_front = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, pidY1, k);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, pidY2, k - 1);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyY_diag_left_back += parallelGridCore->getLatency (pid1, pid2);
      bandwidthY_diag_left_back += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyY_diag_left_back /= tmpCounter;
    bandwidthY_diag_left_back /= tmpCounter;

    if (latencyY_diag_left_back < 0 || bandwidthY_diag_left_back <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyY_diag_left_back = 0;
    bandwidthY_diag_left_back = 0;
  }
#endif

  DOUBLE val = DOUBLE (0);

  ParallelGridCoordinate size = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size;

#ifdef GRID_2D
  DOUBLE sizeY = DOUBLE (size.get1 ());
#endif
#ifdef GRID_3D
  DOUBLE sizeY = DOUBLE (size.get1 () * size.get3 ());
#endif

  if (latencyY >= 0 && bandwidthY > 0)
  {
    val += latencyY + sizeY / bandwidthY;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
#ifdef GRID_2D
  DOUBLE sizeY_diag_left = DOUBLE (1);
  DOUBLE sizeY_diag_right = DOUBLE (1);
#endif
#ifdef GRID_3D
  DOUBLE sizeY_diag_left = DOUBLE (size.get3 ());
  DOUBLE sizeY_diag_right = DOUBLE (size.get3 ());
#endif

  if (latencyY_diag_left >= 0 && latencyY_diag_right >= 0 && bandwidthY_diag_left > 0 && bandwidthY_diag_right > 0)
  {
    val += latencyY_diag_left + sizeY_diag_left / bandwidthY_diag_left
           + latencyY_diag_right + sizeY_diag_right / bandwidthY_diag_right;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE sizeY_diag_back = DOUBLE (size.get1 ());
  DOUBLE sizeY_diag_front = DOUBLE (size.get1 ());

  if (latencyY_diag_back >= 0 && latencyY_diag_front >= 0 && bandwidthY_diag_back > 0 && bandwidthY_diag_front > 0)
  {
    val += latencyY_diag_back + sizeY_diag_back / bandwidthY_diag_back
           + latencyY_diag_front + sizeY_diag_front / bandwidthY_diag_front;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (latencyY_diag_left_back >= 0 && latencyY_diag_left_front >= 0 && latencyY_diag_right_back >= 0 && latencyY_diag_right_front >= 0
      && bandwidthY_diag_left_back > 0 && bandwidthY_diag_left_front > 0 && bandwidthY_diag_right_back > 0 && bandwidthY_diag_right_front > 0)
  {
    val += latencyY_diag_left_back + 1.0 / bandwidthY_diag_left_back
           + latencyY_diag_left_front + 1.0 / bandwidthY_diag_left_front
           + latencyY_diag_right_back + 1.0 / bandwidthY_diag_right_back
           + latencyY_diag_right_front + 1.0 / bandwidthY_diag_right_front;
  }
#endif

  return val;
}

template <SchemeType_t Type, LayoutType layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::estimateBorderY (std::vector<NodeBorder_t> &borders,
                                                           const std::vector<grid_coord> &spreadY) const
{
  int i = 0;

  while (i < parallelGridCore->getNodeGridSizeY () - 1)
  {
    while (i < parallelGridCore->getNodeGridSizeY ()
           && spreadY[i] == 0)
    {
      ++i;
    }

    int j = i + 1;
    while (j < parallelGridCore->getNodeGridSizeY ()
           && spreadY[j] == 0)
    {
      ++j;
    }

    /*
     * Checking border between i and j
     */
    DOUBLE timeVal = estimateTimeAcrossAxisY (i, j);
    if (timeVal == DOUBLE (0))
    {
      /*
       * No time estimation is available
       */
      continue;
    }
    borders.push_back (NodeBorder_t (i, j, OY, timeVal));
  }
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Estimate share time across border between pidX1 and pidX2. This is just estimation, which is used to determine the
 * border with the highest share overhead.
 *
 * @return share time for specified buffer size
 *         0, if share time can't be estimated
 */
template <SchemeType_t Type, LayoutType layout_type>
DOUBLE
ParallelYeeGridLayout<Type, layout_type>::estimateTimeAcrossAxisZ (int pidZ1, /**< z coordinate of first process plane */
                                                                   int pidZ2) const /**< z coordinate of second process plane */
{
  bool enabled1 = false;
  bool enabled2 = false;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i, j, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      enabled1 = true;
    }

    if (parallelGridCore->getNodeState ()[pid2] == 1)
    {
      enabled2 = true;
    }
  }

  if (!enabled1 || !enabled2)
  {
    /*
     * One of two processes is disabled, so, no latency for them
     */
    return 0.0;
  }

  int tmpCounter = 0;

  DOUBLE latencyZ = DOUBLE (0);
  DOUBLE bandwidthZ = DOUBLE (0);

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyZ_diag_right = DOUBLE (0);
  DOUBLE latencyZ_diag_left = DOUBLE (0);
  DOUBLE bandwidthZ_diag_right = DOUBLE (0);
  DOUBLE bandwidthZ_diag_left = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyZ_diag_down = DOUBLE (0);
  DOUBLE latencyZ_diag_up = DOUBLE (0);
  DOUBLE bandwidthZ_diag_down = DOUBLE (0);
  DOUBLE bandwidthZ_diag_up = DOUBLE (0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE latencyZ_diag_left_down = DOUBLE (0);
  DOUBLE latencyZ_diag_left_up = DOUBLE (0);
  DOUBLE latencyZ_diag_right_down = DOUBLE (0);
  DOUBLE latencyZ_diag_right_up = DOUBLE (0);

  DOUBLE bandwidthZ_diag_left_down = DOUBLE (0);
  DOUBLE bandwidthZ_diag_left_up = DOUBLE (0);
  DOUBLE bandwidthZ_diag_right_down = DOUBLE (0);
  DOUBLE bandwidthZ_diag_right_up = DOUBLE (0);
#endif

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i, j, pidZ2);

    ASSERT (parallelGridCore->getNodeState ()[pid1] == parallelGridCore->getNodeState ()[pid2]);
    if (parallelGridCore->getNodeState ()[pid1] == 1)
    {
      latencyZ += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ /= tmpCounter;
    bandwidthZ /= tmpCounter;

    if (latencyZ < 0 || bandwidthZ <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ = 0;
    bandwidthZ = 0;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, j, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_right += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_right += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_right /= tmpCounter;
    bandwidthZ_diag_right /= tmpCounter;

    if (latencyZ_diag_right < 0 || bandwidthZ_diag_right <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_right = 0;
    bandwidthZ_diag_right = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, j, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_left += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_left += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_left /= tmpCounter;
    bandwidthZ_diag_left /= tmpCounter;

    if (latencyZ_diag_left < 0 || bandwidthZ_diag_left <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_left = 0;
    bandwidthZ_diag_left = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i, j + 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_up += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_up += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_up /= tmpCounter;
    bandwidthZ_diag_up /= tmpCounter;

    if (latencyZ_diag_up < 0 || bandwidthZ_diag_up <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_up = 0;
    bandwidthZ_diag_up = 0;
  }

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i, j - 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_down += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_down += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_down /= tmpCounter;
    bandwidthZ_diag_down /= tmpCounter;

    if (latencyZ_diag_down < 0 || bandwidthZ_diag_down <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_down = 0;
    bandwidthZ_diag_down = 0;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_right_up += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_right_up += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_right_up /= tmpCounter;
    bandwidthZ_diag_right_up /= tmpCounter;

    if (latencyZ_diag_right_up < 0 || bandwidthZ_diag_right_up <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_right_up = 0;
    bandwidthZ_diag_right_up = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_left_up += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_left_up += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_left_up /= tmpCounter;
    bandwidthZ_diag_left_up /= tmpCounter;

    if (latencyZ_diag_left_up < 0 || bandwidthZ_diag_left_up <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_left_up = 0;
    bandwidthZ_diag_left_up = 0;
  }

  tmpCounter = 0;

  for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_right_down += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_right_down += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_right_down /= tmpCounter;
    bandwidthZ_diag_right_down /= tmpCounter;

    if (latencyZ_diag_right_down < 0 || bandwidthZ_diag_right_down <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_right_down = 0;
    bandwidthZ_diag_right_down = 0;
  }

  tmpCounter = 0;

  for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
  for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
  {
    int pid1 = parallelGridCore->getNodeGrid (i, j, pidZ1);
    int pid2 = parallelGridCore->getNodeGrid (i - 1, j - 1, pidZ2);

    if (parallelGridCore->getNodeState ()[pid1] == 1
        && parallelGridCore->getNodeState ()[pid2] == 1)
    {
      latencyZ_diag_left_down += parallelGridCore->getLatency (pid1, pid2);
      bandwidthZ_diag_left_down += parallelGridCore->getBandwidth (pid1, pid2);
      ++tmpCounter;
    }
  }
  if (tmpCounter > 0)
  {
    latencyZ_diag_left_down /= tmpCounter;
    bandwidthZ_diag_left_down /= tmpCounter;

    if (latencyZ_diag_left_down < 0 || bandwidthZ_diag_left_down <= 0)
    {
      return 0.0;
    }
  }
  else
  {
    latencyZ_diag_left_down = 0;
    bandwidthZ_diag_left_down = 0;
  }
#endif

  DOUBLE val = DOUBLE (0);

  ParallelGridCoordinate size = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size;

  DOUBLE sizeZ = DOUBLE (size.get1 () * size.get2 ());

  if (latencyZ >= 0 && bandwidthZ > 0)
  {
    val += latencyZ + sizeZ / bandwidthZ;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE sizeZ_diag_left = DOUBLE (size.get2 ());
  DOUBLE sizeZ_diag_right = DOUBLE (size.get2 ());

  if (latencyZ_diag_left >= 0 && latencyZ_diag_right >= 0 && bandwidthZ_diag_left > 0 && bandwidthZ_diag_right > 0)
  {
    val += latencyZ_diag_left + sizeZ_diag_left / bandwidthZ_diag_left
           + latencyZ_diag_right + sizeZ_diag_right / bandwidthZ_diag_right;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  DOUBLE sizeZ_diag_down = DOUBLE (size.get1 ());
  DOUBLE sizeZ_diag_up = DOUBLE (size.get1 ());

  if (latencyZ_diag_down >= 0 && latencyZ_diag_up >= 0 && bandwidthZ_diag_down > 0 && bandwidthZ_diag_up > 0)
  {
    val += latencyZ_diag_down + sizeZ_diag_down / bandwidthZ_diag_down
           + latencyZ_diag_up + sizeZ_diag_up / bandwidthZ_diag_up;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (latencyZ_diag_left_down >= 0 && latencyZ_diag_left_up >= 0 && latencyZ_diag_right_down >= 0 && latencyZ_diag_right_up >= 0
      && bandwidthZ_diag_left_down > 0 && bandwidthZ_diag_left_up > 0 && bandwidthZ_diag_right_down > 0 && bandwidthZ_diag_right_up > 0)
  {
    val += latencyZ_diag_left_down + 1.0 / bandwidthZ_diag_left_down
           + latencyZ_diag_left_up + 1.0 / bandwidthZ_diag_left_up
           + latencyZ_diag_right_down + 1.0 / bandwidthZ_diag_right_down
           + latencyZ_diag_right_up + 1.0 / bandwidthZ_diag_right_up;
  }
#endif

  return val;
}

template <SchemeType_t Type, LayoutType layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::estimateBorderZ (std::vector<NodeBorder_t> &borders,
                                                           const std::vector<grid_coord> &spreadZ) const
{
  int i = 0;

  while (i < parallelGridCore->getNodeGridSizeZ () - 1)
  {
    while (i < parallelGridCore->getNodeGridSizeZ ()
           && spreadZ[i] == 0)
    {
      ++i;
    }

    int j = i + 1;
    while (j < parallelGridCore->getNodeGridSizeZ ()
           && spreadZ[j] == 0)
    {
      ++j;
    }

    /*
     * Checking border between i and j
     */
    DOUBLE timeVal = estimateTimeAcrossAxisZ (i, j);
    if (timeVal == DOUBLE (0))
    {
      /*
       * No time estimation is available
       */
      continue;
    }
    borders.push_back (NodeBorder_t (i, j, OZ, timeVal));
  }
}
#endif

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::estimatePerfLRAxis (int startXL,
                                                                   int endXL,
                                                                   int startXR,
                                                                   int endXR,
#if defined (GRID_2D) || defined (GRID_3D)
                                                                   int startYL,
                                                                   int endYL,
                                                                   int startYR,
                                                                   int endYR,
#endif
#if defined (GRID_3D)
                                                                   int startZL,
                                                                   int endZL,
                                                                   int startZR,
                                                                   int endZR,
#endif
                                                                   DOUBLE &perf_left,
                                                                   DOUBLE &perf_right)
{
  perf_left = 0;
  perf_right = 0;

  for (int i = startXL; i < endXL; ++i)
#if defined (GRID_2D) || defined (GRID_3D)
  for (int j = startYL; j < endYL; ++j)
#endif
#if defined (GRID_3D)
  for (int k = startZL; k < endZL; ++k)
#endif
  {
#ifdef GRID_1D
    int pid = parallelGridCore->getNodeGrid (i);
#endif
#ifdef GRID_2D
    int pid = parallelGridCore->getNodeGrid (i, j);
#endif
#ifdef GRID_3D
    int pid = parallelGridCore->getNodeGrid (i, j, k);
#endif

    if (parallelGridCore->getNodeState ()[pid] == 1)
    {
      perf_left += parallelGridCore->getPerf (pid);
    }
  }

  for (int i = startXR; i < endXR; ++i)
#if defined (GRID_2D) || defined (GRID_3D)
  for (int j = startYR; j < endYR; ++j)
#endif
#if defined (GRID_3D)
  for (int k = startZR; k < endZR; ++k)
#endif
  {
#ifdef GRID_1D
    int pid = parallelGridCore->getNodeGrid (i);
#endif
#ifdef GRID_2D
    int pid = parallelGridCore->getNodeGrid (i, j);
#endif
#ifdef GRID_3D
    int pid = parallelGridCore->getNodeGrid (i, j, k);
#endif

    if (parallelGridCore->getNodeState ()[pid] == 1)
    {
      perf_right += parallelGridCore->getPerf (pid);
    }
  }
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::estimatePerfLR (NodeBorder_t border,
                                                               DOUBLE &perf_left,
                                                               DOUBLE &perf_right)
{
  if (border.axis == OX)
  {
    estimatePerfLRAxis (0, border.pid_coord1 + 1,
                        border.pid_coord2, parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                        0, parallelGridCore->getNodeGridSizeZ (),
                        0, parallelGridCore->getNodeGridSizeZ (),
#endif
                        perf_left, perf_right);
  }
  else if (border.axis == OY)
  {
    estimatePerfLRAxis (0, parallelGridCore->getNodeGridSizeX (),
                        0, parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                        0, border.pid_coord1 + 1,
                        border.pid_coord2, parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                        0, parallelGridCore->getNodeGridSizeZ (),
                        0, parallelGridCore->getNodeGridSizeZ (),
#endif
                        perf_left, perf_right);
  }
  else if (border.axis == OZ)
  {
    estimatePerfLRAxis (0, parallelGridCore->getNodeGridSizeX (),
                        0, parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                        0, border.pid_coord1 + 1,
                        border.pid_coord2, parallelGridCore->getNodeGridSizeZ (),
#endif
                        perf_left, perf_right);
  }
  else
  {
    UNREACHABLE;
  }
}

template <SchemeType_t Type, LayoutType layout_type>
bool ParallelYeeGridLayout<Type, layout_type>::disableAxisLR (NodeBorder_t border,
                                                              std::vector<grid_coord> &spreadX,
                                                              int startXL,
                                                              int endXL,
                                                              int startXR,
                                                              int endXR,
                                                              int sizeX,
#if defined (GRID_2D) || defined (GRID_3D)
                                                              std::vector<grid_coord> &spreadY,
                                                              int startYL,
                                                              int endYL,
                                                              int startYR,
                                                              int endYR,
                                                              int sizeY,
#endif
#if defined (GRID_3D)
                                                              std::vector<grid_coord> &spreadZ,
                                                              int startZL,
                                                              int endZL,
                                                              int startZR,
                                                              int endZR,
                                                              int sizeZ,
#endif
                                                              DOUBLE perf_left,
                                                              DOUBLE perf_right,
                                                              DOUBLE perf_all,
                                                              DOUBLE overallSize)
{
  bool disabled = false;

  DOUBLE max_share_LR_left = DOUBLE (0);
  DOUBLE max_share_DU_left = DOUBLE (0);
  DOUBLE max_share_BF_left = DOUBLE (0);

  DOUBLE max_share_LD_RU_left = DOUBLE (0);
  DOUBLE max_share_LU_RD_left = DOUBLE (0);
  DOUBLE max_share_LB_RF_left = DOUBLE (0);
  DOUBLE max_share_LF_RB_left = DOUBLE (0);
  DOUBLE max_share_DB_UF_left = DOUBLE (0);
  DOUBLE max_share_DF_UB_left = DOUBLE (0);

  DOUBLE max_share_LDB_RUF_left = DOUBLE (0);
  DOUBLE max_share_RDB_LUF_left = DOUBLE (0);
  DOUBLE max_share_RUB_LDF_left = DOUBLE (0);
  DOUBLE max_share_LUB_RDF_left = DOUBLE (0);

  DOUBLE max_share_time_left = DOUBLE (0);
  DOUBLE valueLeft = DOUBLE (0);


  DOUBLE max_share_LR_right = DOUBLE (0);
  DOUBLE max_share_DU_right = DOUBLE (0);
  DOUBLE max_share_BF_right = DOUBLE (0);

  DOUBLE max_share_LD_RU_right = DOUBLE (0);
  DOUBLE max_share_LU_RD_right = DOUBLE (0);
  DOUBLE max_share_LB_RF_right = DOUBLE (0);
  DOUBLE max_share_LF_RB_right = DOUBLE (0);
  DOUBLE max_share_DB_UF_right = DOUBLE (0);
  DOUBLE max_share_DF_UB_right = DOUBLE (0);

  DOUBLE max_share_LDB_RUF_right = DOUBLE (0);
  DOUBLE max_share_RDB_LUF_right = DOUBLE (0);
  DOUBLE max_share_RUB_LDF_right = DOUBLE (0);
  DOUBLE max_share_LUB_RDF_right = DOUBLE (0);

  DOUBLE max_share_time_right = DOUBLE (0);
  DOUBLE valueRight = DOUBLE (0);


  DOUBLE max_share_LR_all = DOUBLE (0);
  DOUBLE max_share_DU_all = DOUBLE (0);
  DOUBLE max_share_BF_all = DOUBLE (0);

  DOUBLE max_share_LD_RU_all = DOUBLE (0);
  DOUBLE max_share_LU_RD_all = DOUBLE (0);
  DOUBLE max_share_LB_RF_all = DOUBLE (0);
  DOUBLE max_share_LF_RB_all = DOUBLE (0);
  DOUBLE max_share_DB_UF_all = DOUBLE (0);
  DOUBLE max_share_DF_UB_all = DOUBLE (0);

  DOUBLE max_share_LDB_RUF_all = DOUBLE (0);
  DOUBLE max_share_RDB_LUF_all = DOUBLE (0);
  DOUBLE max_share_RUB_LDF_all = DOUBLE (0);
  DOUBLE max_share_LUB_RDF_all = DOUBLE (0);

  DOUBLE max_share_time_all = DOUBLE (0);
  DOUBLE valueAll = DOUBLE (0);

  {
    std::vector<grid_coord> tmp_spread;

    std::vector<grid_coord> *tmp_spreadX = NULLPTR;
#if defined (GRID_2D) || defined (GRID_3D)
    std::vector<grid_coord> *tmp_spreadY = NULLPTR;
#endif
#if defined (GRID_3D)
    std::vector<grid_coord> *tmp_spreadZ = NULLPTR;
#endif

    if (border.axis == OX)
    {
      tmp_spread.resize (sizeX);
      tmp_spreadX = &tmp_spread;
#if defined (GRID_2D) || defined (GRID_3D)
      tmp_spreadY = &spreadY;
#endif
#if defined (GRID_3D)
      tmp_spreadZ = &spreadZ;
#endif

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startXL, endXL, OX
#if defined (GRID_2D) || defined (GRID_3D)
                               , startYL, endYL, OY
#endif
#if defined (GRID_3D)
                               , startZL, endZL, OZ
#endif
                               );
    }
#if defined (GRID_2D) || defined (GRID_3D)
    else if (border.axis == OY)
    {
      tmp_spread.resize (sizeY);
      tmp_spreadX = &spreadX;
      tmp_spreadY = &tmp_spread;
#if defined (GRID_3D)
      tmp_spreadZ = &spreadZ;
#endif

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startYL, endYL, OY,
                               startXL, endXL, OX
#if defined (GRID_3D)
                               , startZL, endZL, OZ
#endif
                               );
    }
#endif
#if defined (GRID_3D)
    else if (border.axis == OZ)
    {
      tmp_spread.resize (sizeZ);
      tmp_spreadX = &spreadX;
      tmp_spreadY = &spreadY;
      tmp_spreadZ = &tmp_spread;

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startZL, endZL, OZ,
                               startXL, endXL, OX,
                               startYL, endYL, OY);
    }
#endif
    else
    {
      UNREACHABLE;
    }


    findMaxTimes (max_share_LR_left, max_share_DU_left, max_share_BF_left,
                  max_share_LD_RU_left, max_share_LU_RD_left, max_share_LB_RF_left, max_share_LF_RB_left,
                  max_share_DB_UF_left, max_share_DF_UB_left,
                  max_share_LDB_RUF_left, max_share_RDB_LUF_left, max_share_RUB_LDF_left, max_share_LUB_RDF_left,
                  *tmp_spreadX, startXL, endXL
#if defined (GRID_2D) || defined (GRID_3D)
                  , *tmp_spreadY, startYL, endYL
#endif
#if defined (GRID_3D)
                  , *tmp_spreadZ, startZL, endZL
#endif
                  );

    max_share_time_left = 2 * (max_share_LR_left + max_share_DU_left + max_share_BF_left
                               + max_share_LD_RU_left + max_share_LU_RD_left + max_share_LB_RF_left + max_share_LF_RB_left
                               + max_share_DB_UF_left + max_share_DF_UB_left
                               + max_share_LDB_RUF_left + max_share_RDB_LUF_left + max_share_RUB_LDF_left + max_share_LUB_RDF_left);

    valueLeft = overallSize / perf_left + max_share_time_left;
  }

  {
    std::vector<grid_coord> tmp_spread;

    std::vector<grid_coord> *tmp_spreadX = NULLPTR;
#if defined (GRID_2D) || defined (GRID_3D)
    std::vector<grid_coord> *tmp_spreadY = NULLPTR;
#endif
#if defined (GRID_3D)
    std::vector<grid_coord> *tmp_spreadZ = NULLPTR;
#endif

    if (border.axis == OX)
    {
      tmp_spread.resize (sizeX);
      tmp_spreadX = &tmp_spread;
#if defined (GRID_2D) || defined (GRID_3D)
      tmp_spreadY = &spreadY;
#endif
#if defined (GRID_3D)
      tmp_spreadZ = &spreadZ;
#endif

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startXR, endXR, OX
#if defined (GRID_2D) || defined (GRID_3D)
                               , startYR, endYR, OY
#endif
#if defined (GRID_3D)
                               , startZR, endZR, OZ
#endif
                               );
    }
#if defined (GRID_2D) || defined (GRID_3D)
    else if (border.axis == OY)
    {
      tmp_spread.resize (sizeY);
      tmp_spreadX = &spreadX;
      tmp_spreadY = &tmp_spread;
#if defined (GRID_3D)
      tmp_spreadZ = &spreadZ;
#endif

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startYR, endYR, OY,
                               startXR, endXR, OX
#if defined (GRID_3D)
                               , startZR, endZR, OZ
#endif
                               );
    }
#endif
#if defined (GRID_3D)
    else if (border.axis == OZ)
    {
      tmp_spread.resize (sizeZ);
      tmp_spreadX = &spreadX;
      tmp_spreadY = &spreadY;
      tmp_spreadZ = &tmp_spread;

      spreadGridPointsPerAxis (tmp_spread, perf_all,
                               startZR, endZR, OZ,
                               startXR, endXR, OX,
                               startYR, endYR, OY);
    }
#endif
    else
    {
      UNREACHABLE;
    }

    findMaxTimes (max_share_LR_left, max_share_DU_left, max_share_BF_left,
                  max_share_LD_RU_left, max_share_LU_RD_left, max_share_LB_RF_left, max_share_LF_RB_left,
                  max_share_DB_UF_left, max_share_DF_UB_left,
                  max_share_LDB_RUF_left, max_share_RDB_LUF_left, max_share_RUB_LDF_left, max_share_LUB_RDF_left,
                  *tmp_spreadX, startXR, endXR
#if defined (GRID_2D) || defined (GRID_3D)
                  , *tmp_spreadY, startYR, endYR
#endif
#if defined (GRID_3D)
                  , *tmp_spreadZ, startZR, endZR
#endif
                  );

    max_share_time_right = 2 * (max_share_LR_right + max_share_DU_right + max_share_BF_right
                               + max_share_LD_RU_right + max_share_LU_RD_right + max_share_LB_RF_right + max_share_LF_RB_right
                               + max_share_DB_UF_right + max_share_DF_UB_right
                               + max_share_LDB_RUF_right + max_share_RDB_LUF_right + max_share_RUB_LDF_right + max_share_LUB_RDF_right);

    valueRight = overallSize / perf_right + max_share_time_right;
  }

  findMaxTimes (max_share_LR_all, max_share_DU_all, max_share_BF_all,
                max_share_LD_RU_all, max_share_LU_RD_all, max_share_LB_RF_all, max_share_LF_RB_all,
                max_share_DB_UF_all, max_share_DF_UB_all,
                max_share_LDB_RUF_all, max_share_RDB_LUF_all, max_share_RUB_LDF_all, max_share_LUB_RDF_all,
                spreadX, 0, parallelGridCore->getNodeGridSizeX ()
#if defined (GRID_2D) || defined (GRID_3D)
                , spreadY, 0, parallelGridCore->getNodeGridSizeY ()
#endif
#if defined (GRID_3D)
                , spreadZ, 0, parallelGridCore->getNodeGridSizeZ ()
#endif
                );

  max_share_time_all = 2 * (max_share_LR_all + max_share_DU_all + max_share_BF_all
                            + max_share_LD_RU_all + max_share_LU_RD_all + max_share_LB_RF_all + max_share_LF_RB_all
                            + max_share_DB_UF_all + max_share_DF_UB_all
                            + max_share_LDB_RUF_all + max_share_RDB_LUF_all + max_share_RUB_LDF_all + max_share_LUB_RDF_all);

  valueAll = overallSize / perf_all + max_share_time_all;

  DOUBLE new_perf_all;

  if (parallelGridCore->getProcessId () == 0)
  {
    printf ("# %d (%d <--> %d) =========== %f %f %f =======\n",
      border.axis, border.pid_coord1, border.pid_coord2, valueLeft, valueRight, valueAll);
  }

  if (valueLeft < valueAll && valueLeft < valueRight)
  {
    // disable right
    printf ("DISABLE RIGHT\n");
    disabled = true;

    for (int i = startXR; i < endXR; ++i)
#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = startYR; j < endYR; ++j)
#endif
#if defined (GRID_3D)
    for (int k = startZR; k < endZR; ++k)
#endif
    {
#ifdef GRID_1D
      int pid = parallelGridCore->getNodeGrid (i);
#endif
#ifdef GRID_2D
      int pid = parallelGridCore->getNodeGrid (i, j);
#endif
#ifdef GRID_3D
      int pid = parallelGridCore->getNodeGrid (i, j, k);
#endif

      parallelGridCore->getNodeState ()[pid] = 0;
    }

    new_perf_all = perf_left;
  }
  else if (valueRight < valueAll && valueRight < valueLeft)
  {
    // disable left
    printf ("DISABLE LEFT\n");
    disabled = true;

    for (int i = startXL; i < endXL; ++i)
#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = startYL; j < endYL; ++j)
#endif
#if defined (GRID_3D)
    for (int k = startZL; k < endZL; ++k)
#endif
    {
#ifdef GRID_1D
      int pid = parallelGridCore->getNodeGrid (i);
#endif
#ifdef GRID_2D
      int pid = parallelGridCore->getNodeGrid (i, j);
#endif
#ifdef GRID_3D
      int pid = parallelGridCore->getNodeGrid (i, j, k);
#endif

      parallelGridCore->getNodeState ()[pid] = 0;
    }

    new_perf_all = perf_right;
  }
  else
  {
    printf ("DECIDED TO BREAK\n");
  }

  if (disabled)
  {
    /*
     * Update the spread if we disabled smth
     */
    // TODO: remove this, as we already computed spread in tmp_spread

    if (border.axis == OX)
    {
      spreadGridPointsPerAxis (spreadX, new_perf_all,
                               0, parallelGridCore->getNodeGridSizeX (), OX
#if defined (GRID_2D) || defined (GRID_3D)
                               , 0, parallelGridCore->getNodeGridSizeY (), OY
#endif
#if defined (GRID_3D)
                               , 0, parallelGridCore->getNodeGridSizeZ (), OZ
#endif
                               );
    }
#if defined (GRID_2D) || defined (GRID_3D)
    else if (border.axis == OY)
    {
      spreadGridPointsPerAxis (spreadY, new_perf_all,
                               0, parallelGridCore->getNodeGridSizeY (), OY,
                               0, parallelGridCore->getNodeGridSizeX (), OX
#if defined (GRID_3D)
                               , 0, parallelGridCore->getNodeGridSizeZ (), OZ
#endif
                               );
    }
#endif
#if defined (GRID_3D)
    else if (border.axis == OZ)
    {
      spreadGridPointsPerAxis (spreadZ, perf_all,
                               0, parallelGridCore->getNodeGridSizeZ (), OZ,
                               0, parallelGridCore->getNodeGridSizeX (), OX,
                               0, parallelGridCore->getNodeGridSizeY (), OY);
    }
#endif
    else
    {
      UNREACHABLE;
    }

    disableNodesAfterSpread (spreadX
#if defined (GRID_2D) || defined (GRID_3D)
                             , spreadY
#endif
#if defined (GRID_3D)
                             , spreadZ
#endif
                             );

    printf ("DECIDED TO BREAK\n");
  }

  return disabled;
}

/**
 * Disable all to left or to right from the border
 *
 * @return true is smth was disabled
 */
template <SchemeType_t Type, LayoutType layout_type>
bool ParallelYeeGridLayout<Type, layout_type>::disableLR (NodeBorder_t border,
                                                          std::vector<grid_coord> &spreadX,
#if defined (GRID_2D) || defined (GRID_3D)
                                                          std::vector<grid_coord> &spreadY,
#endif
#if defined (GRID_3D)
                                                          std::vector<grid_coord> &spreadZ,
#endif
                                                          DOUBLE perf_left,
                                                          DOUBLE perf_right)
{
  bool disabled = false;

  DOUBLE perf_all = perf_left + perf_right;

  /*
   * Get total number of grid points
   */
  DOUBLE overallSize = (DOUBLE) YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.calculateTotalCoord ();

  if (border.axis == OX)
  {
    disabled = disableAxisLR (border,
                   spreadX, 0, border.pid_coord1 + 1, border.pid_coord2, parallelGridCore->getNodeGridSizeX (), parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                   spreadY, 0, parallelGridCore->getNodeGridSizeY (), 0, parallelGridCore->getNodeGridSizeY (), parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                   spreadZ, 0, parallelGridCore->getNodeGridSizeZ (), 0, parallelGridCore->getNodeGridSizeZ (), parallelGridCore->getNodeGridSizeZ (),
#endif
                   perf_left, perf_right, perf_all, overallSize);
  }
  else if (border.axis == OY)
  {
    disabled = disableAxisLR (border,
                   spreadX, 0, parallelGridCore->getNodeGridSizeX (), 0, parallelGridCore->getNodeGridSizeX (), parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                   spreadY, 0, border.pid_coord1 + 1, border.pid_coord2, parallelGridCore->getNodeGridSizeY (), parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                   spreadZ, 0, parallelGridCore->getNodeGridSizeZ (), 0, parallelGridCore->getNodeGridSizeZ (), parallelGridCore->getNodeGridSizeZ (),
#endif
                   perf_left, perf_right, perf_all, overallSize);
  }
  else if (border.axis == OZ)
  {
    disabled = disableAxisLR (border,
                   spreadX, 0, parallelGridCore->getNodeGridSizeX (), 0, parallelGridCore->getNodeGridSizeX (), parallelGridCore->getNodeGridSizeX (),
#if defined (GRID_2D) || defined (GRID_3D)
                   spreadY, 0, parallelGridCore->getNodeGridSizeY (), 0, parallelGridCore->getNodeGridSizeY (), parallelGridCore->getNodeGridSizeY (),
#endif
#if defined (GRID_3D)
                   spreadZ, 0, border.pid_coord1 + 1, border.pid_coord2, parallelGridCore->getNodeGridSizeZ (), parallelGridCore->getNodeGridSizeZ (),
#endif
                   perf_left, perf_right, perf_all, overallSize);
  }

  return disabled;
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::checkDisablingConditions (std::vector<grid_coord> &spreadX
#if defined (GRID_2D) || defined (GRID_3D)
                                                                         , std::vector<grid_coord> &spreadY
#endif
#if defined (GRID_3D)
                                                                         , std::vector<grid_coord> &spreadZ
#endif
                                                                         )
{
  bool flag = true;
  while (flag)
  {
    std::vector<NodeBorder_t> borders;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    estimateBorderX (borders, spreadX);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    estimateBorderY (borders, spreadY);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    estimateBorderZ (borders, spreadZ);
#endif

    if (borders.empty ())
    {
      printf ("Empty borders\n");
      break;
    }

    std::sort (borders.begin (), borders.end ());

    for (int index = borders.size () - 1; index >= 0 && flag; --index)
    {
      NodeBorder_t entry = borders[index];
      printf ("!! BORDER %d : axis %d, (coord1 %d <--> coord2 %d), val %f\n", index, entry.axis, entry.pid_coord1, entry.pid_coord2, entry.val);
    }

    for (int index = borders.size () - 1; index >= 0 && flag; --index)
    {
      /*
       * Try to remove the connection with the highest overhead
       */
      NodeBorder_t entry = borders[index];

      DOUBLE perf_left;
      DOUBLE perf_right;

      estimatePerfLR (entry, perf_left, perf_right);

      flag = !disableLR (entry, spreadX,
#if defined (GRID_2D) || defined (GRID_3D)
                         spreadY,
#endif
#if defined (GRID_3D)
                         spreadZ,
#endif
                         perf_left, perf_right);
    }
  }
}

/**
 * Spread grid points across all axes, across all enabled computational nodes
 */
template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::spreadGridPoints (std::vector<grid_coord> &spreadX
#if defined (GRID_2D) || defined (GRID_3D)
                                                                 , std::vector<grid_coord> &spreadY
#endif
#if defined (GRID_3D)
                                                                 , std::vector<grid_coord> &spreadZ
#endif
                                                                 , DOUBLE sumSpeedEnabled)
{
#ifdef GRID_1D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  spreadGridPointsPerAxis (spreadX, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX);
#endif

  disableNodesAfterSpread (spreadX);
#endif

#ifdef GRID_2D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  spreadGridPointsPerAxis (spreadX, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  spreadGridPointsPerAxis (spreadY, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX);
#endif

  disableNodesAfterSpread (spreadX, spreadY);
#endif

#ifdef GRID_3D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  spreadGridPointsPerAxis (spreadX, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  spreadGridPointsPerAxis (spreadY, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  spreadGridPointsPerAxis (spreadZ, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY);
#endif

  disableNodesAfterSpread (spreadX, spreadY, spreadZ);
#endif
}

/**
 * Check if computational nodes should be disabled based on the spread result
 */
template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::disableNodesAfterSpread (const std::vector<grid_coord> &spreadX
#if defined (GRID_2D) || defined (GRID_3D)
                                                                        , const std::vector<grid_coord> &spreadY
#endif
#if defined (GRID_3D)
                                                                        , const std::vector<grid_coord> &spreadZ
#endif
                                                                       )
{
  for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
#if defined (GRID_2D) || defined (GRID_3D)
  for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
#endif
#if defined (GRID_3D)
  for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
#endif
  {
    bool doDisable = spreadX[i] == 0
#if defined (GRID_2D) || defined (GRID_3D)
                     || spreadY[i] == 0
#endif
#if defined (GRID_3D)
                     || spreadZ[i] == 0
#endif
                     ;

    if (doDisable)
    {
#ifdef GRID_1D
      int pid = parallelGridCore->getNodeGrid (i);
#endif
#ifdef GRID_2D
      int pid = parallelGridCore->getNodeGrid (i, j);
#endif
#ifdef GRID_3D
      int pid = parallelGridCore->getNodeGrid (i, j, k);
#endif

      parallelGridCore->getNodeState ()[pid] = 0;
    }
  }
}

/**
 * Spread grid points across all enabled computational nodes across the chosen axis 1
 */
template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::spreadGridPointsPerAxis (std::vector<grid_coord> &spread,
                                                                        DOUBLE sumSpeedEnabled,
                                                                        int axisStart1,
                                                                        int axisSize1,
                                                                        Axis_t axis1
#if defined (GRID_2D) || defined (GRID_3D)
                                                                        , int axisStart2
                                                                        , int axisSize2
                                                                        , Axis_t axis2
#endif
#if defined (GRID_3D)
                                                                        , int axisStart3
                                                                        , int axisSize3
                                                                        , Axis_t axis3
#endif
                                                                        )
{
#if defined (GRID_2D) || defined (GRID_3D)
  ASSERT (axis1 != axis2);
#endif
#if defined (GRID_3D)
  ASSERT (axis1 != axis2 && axis2 != axis3 && axis1 != axis3);
#endif

  grid_coord sum_spread = 0;

  grid_coord totalSize;
  if (axis1 == OX)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  }
#if defined (GRID_2D) || defined (GRID_3D)
  else if (axis1 == OY)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  }
#endif
#if defined (GRID_3D)
  else if (axis1 == OZ)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
  }
#endif
  else
  {
    UNREACHABLE;
  }

  for (int processAxis1 = axisStart1; processAxis1 < axisSize1; ++processAxis1)
  {
    bool enabled = false;
    DOUBLE sumSpeedEnabledAxis = DOUBLE (0);

#if defined (GRID_2D) || defined (GRID_3D)
    for (int processAxis2 = axisStart2; processAxis2 < axisSize2; ++processAxis2)
#endif
#if defined (GRID_3D)
    for (int processAxis3 = axisStart3; processAxis3 < axisSize3; ++processAxis3)
#endif
    {
      int processX;
#if defined (GRID_2D) || defined (GRID_3D)
      int processY;
#endif
#if defined (GRID_3D)
      int processZ;
#endif

      if (axis1 == OX)
      {
        processX = processAxis1;
      }
#if defined (GRID_2D) || defined (GRID_3D)
      else if (axis1 == OY)
      {
        processY = processAxis1;
      }
#endif
#if defined (GRID_3D)
      else if (axis1 == OZ)
      {
        processZ = processAxis1;
      }
#endif
      else
      {
        UNREACHABLE;
      }

#if defined (GRID_2D) || defined (GRID_3D)
      if (axis2 == OX)
      {
        processX = processAxis2;
      }
      else if (axis2 == OY)
      {
        processY = processAxis2;
      }
#if defined (GRID_3D)
      else if (axis2 == OZ)
      {
        processZ = processAxis2;
      }
#endif
      else
      {
        UNREACHABLE
      }
#endif

#if defined (GRID_3D)
      if (axis3 == OX)
      {
        processX = processAxis3;
      }
      else if (axis3 == OY)
      {
        processY = processAxis3;
      }
      else if (axis3 == OZ)
      {
        processZ = processAxis3;
      }
      else
      {
        UNREACHABLE;
      }
#endif

#ifdef GRID_1D
      int pid = parallelGridCore->getNodeGrid (processX);
#endif
#ifdef GRID_2D
      int pid = parallelGridCore->getNodeGrid (processX, processY);
#endif
#ifdef GRID_3D
      int pid = parallelGridCore->getNodeGrid (processX, processY, processZ);
#endif

      if (parallelGridCore->getNodeState ()[pid] == 1)
      {
        enabled = true;
        sumSpeedEnabledAxis += parallelGridCore->getPerf (pid);
      }
    }

    spread[processAxis1] = ((DOUBLE) totalSize) * sumSpeedEnabledAxis / (sumSpeedEnabled);

    if (spread[processAxis1] < 1
        || !enabled)
    {
      spread[processAxis1] = 0;
    }

    sum_spread += spread[processAxis1];
  }

  grid_coord diff = totalSize - sum_spread;
  int i = 0;
  while (diff > 0)
  {
    if (spread[i] > 0)
    {
      spread[i]++;
      diff--;
    }

    ++i;
    if (i == axisSize1)
    {
      i = 0;
    }
  }
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::findMaxTimes (DOUBLE &max_share_LR,
                                                             DOUBLE &max_share_DU,
                                                             DOUBLE &max_share_BF,
                                                             DOUBLE &max_share_LD_RU,
                                                             DOUBLE &max_share_LU_RD,
                                                             DOUBLE &max_share_LB_RF,
                                                             DOUBLE &max_share_LF_RB,
                                                             DOUBLE &max_share_DB_UF,
                                                             DOUBLE &max_share_DF_UB,
                                                             DOUBLE &max_share_LDB_RUF,
                                                             DOUBLE &max_share_RDB_LUF,
                                                             DOUBLE &max_share_RUB_LDF,
                                                             DOUBLE &max_share_LUB_RDF,
                                                             const std::vector<grid_coord> &spreadX,
                                                             int axisStartX,
                                                             int axisSizeX
#if defined (GRID_2D) || defined (GRID_3D)
                                                             , const std::vector<grid_coord> &spreadY
                                                             , int axisStartY
                                                             , int axisSizeY
#endif
#if defined (GRID_3D)
                                                             , const std::vector<grid_coord> &spreadZ
                                                             , int axisStartZ
                                                             , int axisSizeZ
#endif
                                                             )
{
  for (int i = axisStartX; i < axisSizeX - 1; ++i)
  {
#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = axisStartY; j < axisSizeY; ++j)
#endif
#if defined (GRID_3D)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
#endif
    {
#ifdef GRID_1D
      DOUBLE size = DOUBLE (1);
      int pid1 = parallelGridCore->getNodeGrid (i);
      int pid2 = parallelGridCore->getNodeGrid (i + 1);
#endif
#ifdef GRID_2D
      DOUBLE size = DOUBLE (spreadY[j]);
      int pid1 = parallelGridCore->getNodeGrid (i, j);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j);
#endif
#ifdef GRID_3D
      DOUBLE size = DOUBLE (spreadY[j] * spreadZ[k]);
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k);
#endif

      DOUBLE time_LR = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (size) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LR > max_share_LR)
      {
        max_share_LR = time_LR;
      }
    }

#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = axisStartY; j < axisSizeY - 1; ++j)
#if defined (GRID_3D)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
#endif
    {
#ifdef GRID_2D
      DOUBLE size = DOUBLE (1);
      int pid1 = parallelGridCore->getNodeGrid (i, j);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1);
#endif
#ifdef GRID_3D
      DOUBLE size = DOUBLE (spreadZ[k]);
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k);
#endif

      DOUBLE time_LD_RU = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (size) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LD_RU > max_share_LD_RU)
      {
        max_share_LD_RU = time_LD_RU;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
#if defined (GRID_3D)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
#endif
    {
#ifdef GRID_2D
      DOUBLE size = DOUBLE (1);
      int pid1 = parallelGridCore->getNodeGrid (i, j);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1);
#endif
#ifdef GRID_3D
      DOUBLE size = DOUBLE (spreadZ[k]);
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k);
#endif

      DOUBLE time_LU_RD = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (size) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LU_RD > max_share_LU_RD)
      {
        max_share_LU_RD = time_LU_RD;
      }
    }
#endif

#if defined (GRID_3D)
    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k + 1);

      DOUBLE time_LB_RF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (spreadY[j]) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LB_RF > max_share_LB_RF)
      {
        max_share_LB_RF = time_LB_RF;
      }
    }

    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ + 1; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k - 1);

      DOUBLE time_LF_RB = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (spreadY[j]) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LF_RB > max_share_LF_RB)
      {
        max_share_LF_RB = time_LF_RB;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k + 1);

      DOUBLE time_LDB_RUF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (1) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LDB_RUF > max_share_LDB_RUF)
      {
        max_share_LDB_RUF = time_LDB_RUF;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k + 1);

      DOUBLE time_LUB_RDF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (1) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_LUB_RDF > max_share_LUB_RDF)
      {
        max_share_LUB_RDF = time_LUB_RDF;
      }
    }
#endif
  }

  for (int i = axisStartX; i < axisSizeX; ++i)
  {
#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = axisStartY; j < axisSizeY - 1; ++j)
#if defined (GRID_3D)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
#endif
    {
#ifdef GRID_2D
      DOUBLE size = DOUBLE (spreadX[i]);
      int pid1 = parallelGridCore->getNodeGrid (i, j);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1);
#endif
#ifdef GRID_3D
      DOUBLE size = DOUBLE (spreadX[i] * spreadZ[k]);
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k);
#endif

      DOUBLE time_DU = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (size) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_DU > max_share_DU)
      {
        max_share_DU = time_DU;
      }
    }
#endif

#if defined (GRID_3D)
    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j, k + 1);

      DOUBLE time_BF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (spreadX[i] * spreadY[j]) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_BF > max_share_BF)
      {
        max_share_BF = time_BF;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k + 1);

      DOUBLE time_DB_UF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (spreadX[i]) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_DB_UF > max_share_DB_UF)
      {
        max_share_DB_UF = time_DB_UF;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ + 1; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k - 1);

      DOUBLE time_DF_UB = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (spreadX[i]) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_DF_UB > max_share_DF_UB)
      {
        max_share_DF_UB = time_DF_UB;
      }
    }
  }

  for (int i = axisStartX + 1; i < axisSizeX; ++i)
  {
    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k + 1);

      DOUBLE time_RDB_LUF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (1) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_RDB_LUF > max_share_RDB_LUF)
      {
        max_share_RDB_LUF = time_RDB_LUF;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i - 1, j - 1, k + 1);

      DOUBLE time_RUB_LDF = parallelGridCore->getLatency (pid1, pid2) + DOUBLE (1) / parallelGridCore->getBandwidth (pid1, pid2);
      if (time_RUB_LDF > max_share_RUB_LDF)
      {
        max_share_RUB_LDF = time_RUB_LDF;
      }
    }
#endif
  }
}

/**
 * Rebalance total grid size between notes
 *
 * @return true if spread was changed
 *         false, otherwise
 */
template <SchemeType_t Type, LayoutType layout_type>
bool ParallelYeeGridLayout<Type, layout_type>::Rebalance (time_step difft) /**< number of time steps elapsed since the last rebalance */
{
  ParallelGridCoordinate newSize = sizeForCurNode;
  ParallelGridCoordinate oldSize = sizeForCurNode;

  /*
   * ==== Calculate total perf, latency and bandwidth ====
   */
  DOUBLE sumSpeedEnabled = parallelGridCore->calcTotalPerf (difft);
  parallelGridCore->calcTotalLatencyAndBandwidth (difft);

  /*
   * ==== Spread ====
   */
  std::vector<grid_coord> spreadX (parallelGridCore->getNodeGridSizeX ());
#if defined (GRID_2D) || defined (GRID_3D)
  std::vector<grid_coord> spreadY (parallelGridCore->getNodeGridSizeY ());
#endif
#if defined (GRID_3D)
  std::vector<grid_coord> spreadZ (parallelGridCore->getNodeGridSizeZ ());
#endif

#ifdef GRID_1D
  spreadGridPoints (spreadX, sumSpeedEnabled);
#endif
#ifdef GRID_2D
  spreadGridPoints (spreadX, spreadY, sumSpeedEnabled);
#endif
#ifdef GRID_3D
  spreadGridPoints (spreadX, spreadY, spreadZ, sumSpeedEnabled);
#endif

  /*
   * ==== Check if some nodes are better to be disabled
   */
  if (SOLVER_SETTINGS.getDoCheckDisablingConditions ())
  {
#ifdef GRID_1D
    checkDisablingConditions (spreadX);
#endif
#ifdef GRID_2D
    checkDisablingConditions (spreadX, spreadY);
#endif
#ifdef GRID_3D
    checkDisablingConditions (spreadX, spreadY, spreadZ);
#endif
  }

  /*
   * ==== Check if some nodes are better to be enabled
   */
  if (SOLVER_SETTINGS.getDoCheckEnablingConditions ())
  {
    // TODO: add enabling conditions check
  }

  /*
   * ==== Get current node's chunk size ====
   */

  grid_coord x = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord y = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
#endif
#if defined (GRID_3D)
  grid_coord z = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  x = spreadX[parallelGridCore->getNodeGridX ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  y = spreadY[parallelGridCore->getNodeGridY ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  z = spreadZ[parallelGridCore->getNodeGridZ ()];
#endif

#ifdef GRID_1D
  printf ("#%d state=%d x=%d speed=%f (perfpoints=%f, perftimes=%f) totalX=%f difft=%u sumSpeedEnabled=%f\n",
          parallelGridCore->getProcessId (),
          parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()],
          x,
          parallelGridCore->getPerf (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfPointsPerProcess (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfTimePerProcess (parallelGridCore->getProcessId ()),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
          difft,
          sumSpeedEnabled);
#endif
#ifdef GRID_2D
  printf ("#%d state=%d x=%d y=%d speed=%f (perfpoints=%f, perftimes=%f) totalX=%f totalY=%f difft=%u sumSpeedEnabled=%f\n",
          parallelGridCore->getProcessId (),
          parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()],
          x, y,
          parallelGridCore->getPerf (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfPointsPerProcess (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfTimePerProcess (parallelGridCore->getProcessId ()),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
          difft,
          sumSpeedEnabled);
#endif
#ifdef GRID_3D
  printf ("#%d state=%d x=%d y=%d z=%d speed=%f (perfpoints=%f, perftimes=%f) totalX=%f totalY=%f totalZ=%f difft=%u sumSpeedEnabled=%f\n",
          parallelGridCore->getProcessId (),
          parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()],
          x, y, z,
          parallelGridCore->getPerf (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfPointsPerProcess (parallelGridCore->getProcessId ()),
          parallelGridCore->getTotalSumPerfTimePerProcess (parallelGridCore->getProcessId ()),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
          (DOUBLE)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 (),
          difft,
          sumSpeedEnabled);
#endif

  if (parallelGridCore->getProcessId () == 0)
  {
    for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
    for (int j = 0; j < parallelGridCore->getTotalProcCount (); ++j)
    {
      printf ("Share: %d--%d (latency=%.15f (%f / %f), bw=%.15f (%f / %f))\n",
        i, j,
        parallelGridCore->getLatency (i, j),
        parallelGridCore->getTotalSumLatencyPerConnection (i, j),
        parallelGridCore->getTotalSumLatencyCountPerConnection (i, j),
        parallelGridCore->getBandwidth (i, j),
        parallelGridCore->getTotalSumBandwidthPerConnection (i, j),
        parallelGridCore->getTotalSumBandwidthCountPerConnection (i, j));
    }
  }

  newSize.set1 (x);
#if defined (GRID_2D) || defined (GRID_3D)
  newSize.set2 (y);
#endif
#ifdef GRID_3D
  newSize.set3 (z);
#endif

  sizeForCurNode = newSize;

  /*
   * ==== Clear current clocks and counter with number of points ====
   */

  parallelGridCore->ClearCalcClocks ();
  parallelGridCore->ClearShareClocks ();

  /*
   * ==== Set new counters for number of points for the next between rebalance episode ====
   */
  if (parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()] == 1)
  {
    InitializeCounters ();
  }

  return true;
} /* ParallelYeeGridLayout::Rebalance */

template <SchemeType_t Type, LayoutType layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::InitializeCounters ()
{
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, 1);
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, 1);
  }
#endif
#endif

#ifdef GRID_2D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, sizeForCurNode.get2 ());
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, sizeForCurNode.get2 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  int dirD = parallelGridCore->getNodeForDirection (DOWN);
  if (dirD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirD, sizeForCurNode.get1 ());
  }
  int dirU = parallelGridCore->getNodeForDirection (UP);
  if (dirU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirU, sizeForCurNode.get1 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  int dirLD = parallelGridCore->getNodeForDirection (LEFT_DOWN);
  if (dirLD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLD, 1);
  }
  int dirRD = parallelGridCore->getNodeForDirection (RIGHT_DOWN);
  if (dirRD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRD, 1);
  }

  int dirLU = parallelGridCore->getNodeForDirection (LEFT_UP);
  if (dirLU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLU, 1);
  }
  int dirRU = parallelGridCore->getNodeForDirection (RIGHT_UP);
  if (dirRU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRU, 1);
  }
#endif
#endif

#ifdef GRID_3D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, sizeForCurNode.get2 () * sizeForCurNode.get3 ());
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, sizeForCurNode.get2 () * sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirD = parallelGridCore->getNodeForDirection (DOWN);
  if (dirD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirD, sizeForCurNode.get1 () * sizeForCurNode.get3 ());
  }
  int dirU = parallelGridCore->getNodeForDirection (UP);
  if (dirU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirU, sizeForCurNode.get1 () * sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirB = parallelGridCore->getNodeForDirection (BACK);
  if (dirB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirB, sizeForCurNode.get1 () * sizeForCurNode.get2 ());
  }
  int dirF = parallelGridCore->getNodeForDirection (FRONT);
  if (dirF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirF, sizeForCurNode.get1 () * sizeForCurNode.get2 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLD = parallelGridCore->getNodeForDirection (LEFT_DOWN);
  if (dirLD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLD, sizeForCurNode.get3 ());
  }
  int dirRD = parallelGridCore->getNodeForDirection (RIGHT_DOWN);
  if (dirRD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRD, sizeForCurNode.get3 ());
  }

  int dirLU = parallelGridCore->getNodeForDirection (LEFT_UP);
  if (dirLU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLU, sizeForCurNode.get3 ());
  }
  int dirRU = parallelGridCore->getNodeForDirection (RIGHT_UP);
  if (dirRU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRU, sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirDB = parallelGridCore->getNodeForDirection (DOWN_BACK);
  if (dirDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirDB, sizeForCurNode.get1 ());
  }
  int dirDF = parallelGridCore->getNodeForDirection (DOWN_FRONT);
  if (dirDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirDF, sizeForCurNode.get1 ());
  }

  int dirUB = parallelGridCore->getNodeForDirection (UP_BACK);
  if (dirUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirUB, sizeForCurNode.get1 ());
  }
  int dirUF = parallelGridCore->getNodeForDirection (UP_FRONT);
  if (dirUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirUF, sizeForCurNode.get1 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLB = parallelGridCore->getNodeForDirection (LEFT_BACK);
  if (dirLB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLB, sizeForCurNode.get2 ());
  }
  int dirLF = parallelGridCore->getNodeForDirection (LEFT_FRONT);
  if (dirLF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLF, sizeForCurNode.get2 ());
  }

  int dirRB = parallelGridCore->getNodeForDirection (RIGHT_BACK);
  if (dirRB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRB, sizeForCurNode.get2 ());
  }
  int dirRF = parallelGridCore->getNodeForDirection (RIGHT_FRONT);
  if (dirRF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRF, sizeForCurNode.get2 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLDB = parallelGridCore->getNodeForDirection (LEFT_DOWN_BACK);
  if (dirLDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLDB, 1);
  }
  int dirLDF = parallelGridCore->getNodeForDirection (LEFT_DOWN_FRONT);
  if (dirLDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLDF, 1);
  }

  int dirLUB = parallelGridCore->getNodeForDirection (LEFT_UP_BACK);
  if (dirLUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLUB, 1);
  }
  int dirLUF = parallelGridCore->getNodeForDirection (LEFT_UP_FRONT);
  if (dirLUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLUF, 1);
  }

  int dirRDB = parallelGridCore->getNodeForDirection (RIGHT_DOWN_BACK);
  if (dirRDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRDB, 1);
  }
  int dirRDF = parallelGridCore->getNodeForDirection (RIGHT_DOWN_FRONT);
  if (dirRDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRDF, 1);
  }

  int dirRUB = parallelGridCore->getNodeForDirection (RIGHT_UP_BACK);
  if (dirRUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRUB, 1);
  }
  int dirRUF = parallelGridCore->getNodeForDirection (RIGHT_UP_FRONT);
  if (dirRUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRUF, 1);
  }
#endif
#endif
}

#endif /* DYNAMIC_GRID */

#endif /* PARALLEL_GRID */

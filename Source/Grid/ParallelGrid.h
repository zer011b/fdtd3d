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

#ifndef PARALLEL_GRID_H
#define PARALLEL_GRID_H

#include "ParallelGridCore.h"
#include "ParallelGridGroup.h"

#ifdef PARALLEL_GRID

/**
 * Parallel grid class
 *
 * Dimension of grid is set during build. Virtual topology of grid is also specified during build.
 *
 * Possible combinations:
 *   - 1D grid (has only one dimension and Ox axis):
 *     - Ox axis is spread through all computational nodes. If grid size is N, and number of computational nodes is P,
 *       then grid size per computational node is N/P. In case P is not divider of N, the last node is assigned grid
 *       with size (N - (N/P)*(P - 1)). See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                   ___ ___ ___       ___
 *       Ox axis -> |___|___|___| ... |___|
 *
 *   - 2D grid (has two dimensions and Ox,Oy axis):
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread.
 *       If grid size is N*M, and number of computational nodes is P, then grid size per computational node is (N/P)*M.
 *       In case P is not divider of N, the right border node by Ox axis is assigned grid with size with remainder
 *       (N - (N/P)*(P - 1))*M. See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                   ___ ___ ___       ___
 *                  |   |   |   | ... |   |
 *                  |   |   |   | ... |   |
 *       Ox axis -> |___|___|___| ... |___|

 *     - Ox axis is not spread, Oy axis is spread through all computational nodes.
 *       If grid size is N*M, and number of computational nodes is P, then grid size per computational node is N*(M/P).
 *       In case P is not divider of M, the right border node by Oy axis is assigned grid with size with remainder
 *       N*(M - (M/P)*(P - 1)). See PARALLEL_BUFFER_DIMENSION_1D_Y.
 *                   _____________________
 *                  |_____________________|
 *                  |_____________________|
 *                   .....................
 *                   _____________________
 *       Ox axis -> |_____________________|
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes.
 *       If grid size is N*M, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*(M/Q). In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*(M/Q), in case Q is not divider of M the right border node by Oy axis is assigned
 *       grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1)), in case both P is not divider of N and
 *       Q is not divider of M, the right corner border node by Ox and Oy axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_XY.
 *                   ___ ___ ___       ___
 *                  |___|___|___| ... |___|
 *                  |___|___|___| ... |___|
 *                   .....................
 *                   ___ ___ ___  ...  ___
 *       Ox axis -> |___|___|___| ... |___|
 *
 *   - 3D grid (has three dimensions and Ox,Oy,Oz axis):
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread, Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       (N/P)*M*K. In case P is not divider of N, the right border node by Ox axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*M*K. See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                     ___ ___ ____          ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ / |  ...  /__ / |
 *                  |   |   |   |  |  ... |   |  |
 *                  |   |   |   | /   ... |   | /
 *       Ox axis -> |___|___|___|/    ... |___|/
 *
 *     - Ox axis is not spread, Oy axis is spread through all computational nodes, Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       N*(M/P)*K. In case P is not divider of M, the right border node by Oy axis is assigned grid with size
 *       with remainder N*(M - (M/P)*(P - 1))*K. See PARALLEL_BUFFER_DIMENSION_1D_Y.
 *                     __________________________
 *                    /                         /|
 *                   /________________________ //
 *                  |_________________________|/
 *                   .........................
 *                     __________________________
 *                    /                         /|
 *                   /________________________ //|
 *                  |_________________________|//
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is not spread, Oy axis is not spread, Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       N*(M/P)*K. In case P is not divider of M, the right border node by Oy axis is assigned grid with size
 *       with remainder N*(M - (M/P)*(P - 1))*K. See PARALLEL_BUFFER_DIMENSION_1D_Z.
 *
 *                            __________________________
 *                           /                         /|
 *                          /________________________ / |
 *                         |                         |  |
 *                         | ....................... |  |
 *                       __|_______________________  | /
 *                      /                         /|_|/
 *                     /________________________ / |
 *                    /                         /| |
 *                   /________________________ / | |
 *                  |                         |  |/
 *                  |                         |  /
 *                  |                         | /
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes,
 *       Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*(M/Q)*K. In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*(M/Q)*K, in case Q is not divider of M the right border node by Oy axis is
 *       assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*K, in case both P is not divider of N and
 *       Q is not divider of M the right corner border node by Ox and Oy axis is assigned grid with
 *       size with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*K. See PARALLEL_BUFFER_DIMENSION_2D_XY.
 *                     ___ ___ ____          ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ //|  ...  /__ //|
 *                  |___|___|___|//   ... |___|//
 *                  |___|___|___|/    ... |___|/
 *                   ............................
 *                     ____ ___ ___   ...    ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ //   ...  /__ //
 *       Ox axis -> |___|___|___|/    ... |___|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*M*(K/Q). In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*M*(K/Q), in case Q is not divider of K the right border node by Oz axis is
 *       assigned grid with size with remainder (N/P)*M*(K - (K/Q)*(Q - 1)), in case both P is not divider of N and
 *       Q is not divider of K the right corner border node by Ox and Oz axis is assigned grid with
 *       size with remainder (N - (N/P)*(P - 1))*M*(K - (K/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_XZ.
 *
 *                         ___ ____ ___          ____
 *                        /__ /__ /__ /|    .   /__ /|
 *                       |   |   |   | |   .   |   | |
 *                     __|_ _|_ _|_  | |  .  __|_  | |
 *                    /___/___/___/|_|/  .  /___/|_|/
 *                   /__ /__ /__ /||......./__ /||.
 *                  |   |   |   | ||   .  |   | ||
 *                  |   |   |   | /   .   |   | /
 *       Ox axis -> |___|___|___|/   .    |___|/
 *
 *     - Ox axis is not spread, Oy axis is spread through all computational nodes,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       N*(M/P)*(K/Q). In case P is not divider of M, the right border node by Oy axis is assigned grid with size with
 *       remainder N*(M - (M/P)*(P - 1))*(K/Q), in case Q is not divider of K the right border node by Oz axis is
 *       assigned grid with size with remainder N*(M/P)*(K - (K/Q)*(Q - 1)), in case both P is not divider of M and
 *       Q is not divider of K the right corner border node by Oy and Oz axis is assigned grid with
 *       size with remainder N*(M - (M/P)*(P - 1))*(K - (K/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_YZ.
 *                               __________________________
 *                              /________________________ /|
 *                             |_________________________|/
 *                                                      .
 *                    __________________________ _______.__
 *                   /________________________ /|_______. /|
 *                  |_________________________|/________.|/|
 *                    .        |________________________.|/
 *                    .        .                       .
 *                    .______.__________________     .
 *                   /._______________________ /|  .
 *                  |_________________________|/|.
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q*R, then grid size per computational node is
 *       (N/P)*(M/Q)*(K/R). In case P is not divider of N, the right border node by Ox axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*(M/Q)*(K/R), in case Q is not divider of M the right border node by Oy axis
 *       is assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*(K/R), in case R is not divider of K
 *       the right border node by Oz axis is assigned grid with size with remainder (N/P)*(M/Q)*(K - (K/R)*(R - 1)),
 *       in case both P is not divider of N and Q is not divider of M the right corner border node by Ox and Oy axis is
 *       assigned grid with size with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*(K/R),
 *       in case both P is not divider of N and R is not divider of K the right corner border node by Ox and Oz axis is
 *       assigned grid with size with remainder (N - (N/P)*(P - 1))*(M/Q)*(K - (K/R)*(R - 1)),
 *       in case both Q is not divider of M and R is not divider of K the right corner border node by Oy and Oz axis is
 *       assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*(K - (K/R)*(R - 1)),
 *       in case P is not divider of N, Q is not divider of M and R is not divider of K, the right corner border node
 *       by Ox, Oy and Oz axis is assigned grid with size with remainder
 *       (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*(K - (K/R)*(R - 1)). See PARALLEL_BUFFER_DIMENSION_3D_XYZ.
 *
 * On all pictures above axis are:
 *       Oy    Oz
 *       +    +
 *       |   /
 *       |  /
 *       | /
 *       |/_____________+ Ox
 *
 * All grids corresponding to computational nodes have buffers only on the sides where nodes have neighbors.
 *
 * -------- Coordinate systems --------
 * 1. Total coordinate system is the global coordinate system, considering all computational nodes.
 * 2. Relative coordinate system starts from the start of the chunk (considering buffers!),
 *    stored on this computational node.
 *
 *    border between nodes
 *            *
 *    --------|--------
 *          bb|
 *    --------|--------
 *          *
 *    start of relative coordinate system, considering buffers
 */
class ParallelGrid: public ParallelGridBase
{
private:

  /**
   * Static data shared between all parallel grids on this computational node
   */
  static ParallelGridCore *parallelGridCore;

  /**
   * Static array containing info about all groups
   */
  static std::vector<ParallelGridGroup *> groups;

private:

  /**
   * Index of the parallel group, which stores all the size data
   */
  int groupId;

private:

  void SendRawBuffer (BufferPosition, int);
  void ReceiveRawBuffer (BufferPosition, int);
  void SendReceiveRawBuffer (BufferPosition, int, BufferPosition, int);
  void SendReceiveBuffer (BufferPosition);
  void SendReceive ();

public:

  ParallelGrid (const ParallelGridCoordinate &,
                const ParallelGridCoordinate &,
                time_step,
                const ParallelGridCoordinate &,
                int,
                int,
                const char * = "unnamed");

  virtual ~ParallelGrid () {}

  /**
   * Get parallel group constant
   *
   * @return constant parallel group
   */
  const ParallelGridGroup *getGroupConst () const
  {
    return getGroup (groupId);
  } /* getGroupConst */

  /**
   * Get parallel group
   *
   * @return parallel group
   */
  ParallelGridGroup *getGroup ()
  {
    return getGroup (groupId);
  } /* getGroup */

  void share ();

  /**
   * Get share step
   *
   * @return share step
   */
  time_step getShareStep () const
  {
    return getGroupConst ()->getShareStep ();
  } /* getShareStep */

  /**
   * Get first coordinate from which to perform computations at current step
   *
   * @return first coordinate from which to perform computations at current step
   */
  virtual ParallelGridCoordinate getComputationStart
    (const ParallelGridCoordinate & diffPosStart) const CXX11_OVERRIDE /**< layout coordinate modifier */
  {
    return getGroupConst ()->getComputationStart (diffPosStart);
  } /* getComputationStart */

  /**
   * Get last coordinate until which to perform computations at current step
   *
   * @return last coordinate until which to perform computations at current step
   */
  virtual ParallelGridCoordinate getComputationEnd
    (const ParallelGridCoordinate & diffPosEnd) const CXX11_OVERRIDE /**< layout coordinate modifier */
  {
    return getGroupConst ()->getComputationEnd (diffPosEnd, ParallelGridBase::getSize ());
  } /* ParallelGrid::getComputationEnd */

  /**
   * Get total position in grid from relative position for current computational node
   *
   * @return total position in grid from relative position for current computational node
   */
  virtual ParallelGridCoordinate getTotalPosition (const ParallelGridCoordinate & pos) const CXX11_OVERRIDE /**< relative
                                                                                                     *   position for
                                                                                                     *   current
                                                                                                     *   computational
                                                                                                     *   node */
  {
    ParallelGridCoordinate posStart = getGroupConst ()->getStartPosition ();

    return posStart + pos;
  } /* getTotalPosition */

  /**
   * Get relative position for current computational node from total position
   *
   * @return relative position for current computational node from total position
   */
  virtual ParallelGridCoordinate getRelativePosition (const ParallelGridCoordinate & pos) const CXX11_OVERRIDE /**< total
                                                                                                        *   position in
                                                                                                        *   grid */
  {
    ParallelGridCoordinate posStart = getGroupConst ()->getStartPosition ();

    ASSERT (pos >= posStart);

    return pos - posStart;
  } /* ParallelGrid::getRelativePosition */

  /**
   * Get field value at absolute coordinate in grid
   *
   * @return field value
   */
  virtual FieldValue * getFieldValueByAbsolutePos
    (const ParallelGridCoordinate &absPosition, /**< absolute coordinate in grid */
     int time_step_back) CXX11_OVERRIDE /**< offset in time */
  {
    return getFieldValue (getRelativePosition (absPosition), time_step_back);
  } /* getFieldValueByAbsolutePos */

  /**
   * Get field value at absolute coordinate in grid. If current node does not contain this coordinate, return NULLPTR
   *
   * @return field value or NULLPTR
   */
  virtual FieldValue * getFieldValueOrNullByAbsolutePos
    (const ParallelGridCoordinate &absPosition, /**< absolute coordinate in grid */
     int time_step_back) CXX11_OVERRIDE /**< offset in time */
  {
    if (!hasValueForCoordinate (absPosition))
    {
      return NULLPTR;
    }

    return getFieldValueByAbsolutePos (absPosition, time_step_back);
  } /* getFieldValueOrNullByAbsolutePos */

  virtual FieldValue * getFieldValueCurrentAfterShiftByAbsolutePos
    (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
  {
    return getFieldValue (getRelativePosition (absPosition), 1);
  }

  virtual FieldValue * getFieldValueOrNullCurrentAfterShiftByAbsolutePos
    (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
  {
    if (!hasValueForCoordinate (absPosition))
    {
      return NULLPTR;
    }

    return getFieldValueCurrentAfterShiftByAbsolutePos (absPosition);
  }

  virtual FieldValue * getFieldValuePreviousAfterShiftByAbsolutePos
    (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
  {
    if (gridValues.size () > 2)
    {
      return getFieldValue (getRelativePosition (absPosition), 2);
    }
    else
    {
      return getFieldValue (getRelativePosition (absPosition), 0);
    }
  }

  virtual FieldValue * getFieldValueOrNullPreviousAfterShiftByAbsolutePos
    (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
  {
    if (!hasValueForCoordinate (absPosition))
    {
      return NULLPTR;
    }

    return getFieldValuePreviousAfterShiftByAbsolutePos (absPosition);
  }

  /**
   * Check whether current node has value for coordinate
   *
   * @return true, if current node contains value for coordinate, false otherwise
   */
  bool hasValueForCoordinate (const ParallelGridCoordinate &position) const /**< coordinate of value to check */
  {
    ParallelGridCoordinate posStart = getGroupConst ()->getStartPosition ();
    ParallelGridCoordinate posEnd = posStart + getSize ();

    if (!(position >= posStart)
        || !(position < posEnd))
    {
      return false;
    }

    return true;
  } /* hasValueForCoordinate */

  /**
   * Getter for total size of grid
   *
   * @return total size of grid
   */
  virtual ParallelGridCoordinate getTotalSize () const CXX11_OVERRIDE
  {
    return getGroupConst ()->getTotalSize ();
  } /* getTotalSize */

  ParallelGridBase *gatherFullGrid () const;
  ParallelGridBase *gatherFullGridPlacement (ParallelGridBase *) const;

#ifdef DYNAMIC_GRID
  void Resize (ParallelGridCoordinate);
#endif /* DYNAMIC_GRID */

  /**
   * Getter for start position of chunk, assigned to this computational node (not considering buffers!)
   *
   * @return start position of chunk, assigned to this computational node (not considering buffers!)
   */
  virtual ParallelGridCoordinate getChunkStartPosition () const CXX11_OVERRIDE
  {
    return getGroupConst ()->getChunkStartPosition ();
  } /* getChunkStartPosition */

  /**
   * Check whether position corresponds to left buffer or not
   *
   * @return true, if position is in left buffer, false, otherwise
   */
  virtual bool isBufferLeftPosition (const ParallelGridCoordinate & pos) const CXX11_OVERRIDE /**< position to check */
  {
    ASSERT (pos < getGroupConst ()->getTotalSize ());
    ParallelGridCoordinate chunkStart = getGroupConst ()->getChunkStartPosition ();

    if (pos >= chunkStart)
    {
      return false;
    }
    else
    {
      return true;
    }
  } /* isBufferLeftPosition */

  /**
   * Check whether position corresponds to right buffer or not
   *
   * @return true, if position is in right buffer, false, otherwise
   */
  virtual bool isBufferRightPosition (const ParallelGridCoordinate & pos) const CXX11_OVERRIDE /**< position to check */
  {
    ASSERT (pos < getGroupConst ()->getTotalSize ());
    ParallelGridCoordinate chunkEnd = getGroupConst ()->getChunkStartPosition () + getGroupConst ()->getCurrentSize ();

    if (pos < chunkEnd)
    {
      return false;
    }
    else
    {
      return true;
    }
  } /* isBufferRightPosition */

  BufferPosition getBufferForPosition (ParallelGridCoordinate) const;

  /**
   * Getter for array of coordinate in grid from which to start send values corresponding to direction
   *
   * @return array of coordinate in grid from which to start send values corresponding to direction
   */
  ParallelGridCoordinate getSendStart (int dir) const
  {
    return getGroupConst ()->getSendStart (dir);
  } /* getSendStart */

  /**
   * Getter for array of coordinate in grid until which to send values corresponding to direction
   *
   * @return array of coordinate in grid until which to send values corresponding to direction
   */
  ParallelGridCoordinate getSendEnd (int dir) const
  {
    return getGroupConst ()->getSendEnd (dir);
  } /* getSendEnd */

  /**
   * Getter for array of coordinate in grid from which to start saving received values corresponding to direction
   *
   * @return array of coordinate in grid from which to start saving received values corresponding to direction
   */
  ParallelGridCoordinate getRecvStart (int dir) const
  {
    return getGroupConst ()->getRecvStart (dir);
  } /* getRecvStart */

  /**
   * Getter for array of coordinate in grid until which to save received values corresponding to direction
   *
   * @return array of coordinate in grid until which to save received values corresponding to direction
   */
  ParallelGridCoordinate getRecvEnd (int dir) const
  {
    return getGroupConst ()->getRecvEnd (dir);
  } /* getRecvEnd */

public:

  /**
   * Get parallel grid core
   */
  static ParallelGridCore * getParallelCore ()
  {
    ASSERT (parallelGridCore != NULLPTR);
    return parallelGridCore;
  } /* getParallelCore */

  /**
   * Initialize parallel grid core
   */
  static void initializeParallelCore (ParallelGridCore *core) /**< new parallel grid core */
  {
    ASSERT (parallelGridCore == NULLPTR);

    parallelGridCore = core;

    ParallelGridGroup::initializeParallelCore (parallelGridCore);
  } /* initializeParallelCore */

  /**
   * Get parallel grid group for specified index
   *
   * @return parallel grid group
   */
  static ParallelGridGroup *getGroup (int index) /**< index of parallel grid group */
  {
    ASSERT (index >= 0 && index < groups.size ());
    return groups[index];
  } /* getGroup */

  /**
   * Find parallel grid group with specified characteristics
   *
   * @return index of parallel grid group or -1 if not found
   */
  static int findGroup (const ParallelGridCoordinate &totSize, /**< overall size of grid */
                        const ParallelGridCoordinate &bufSize, /**< buffer size */
                        time_step stepLimit, /**< number of steps before share operations */
                        const ParallelGridCoordinate &curSize, /**< size of grid per node */
                        int storedTimeSteps,
                        int timeOffset) /**< offset of time step in for t+timeOffset/2, at which grid should be shared */
  {
    for (int i = 0; i < groups.size (); ++i)
    {
      ParallelGridGroup *group = getGroup (i);

      if (group->match (totSize, bufSize, stepLimit, curSize, storedTimeSteps, timeOffset))
      {
        /*
         * Found appropriate group
         */
        return i;
      }
    }

    return INVALID_GROUP;
  } /* findGroup */

  /**
   * Add new parallel grid group
   *
   * @return index of group
   */
  static int addGroup (ParallelGridGroup *newgroup) /**< new parallel grid group */
  {
    int index = groups.size ();
    groups.push_back (newgroup);
    return index;
  } /* addGroup */
}; /* ParallelGrid */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_H */

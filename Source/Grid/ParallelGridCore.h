#ifndef PARALLEL_GRID_CORE_H
#define PARALLEL_GRID_CORE_H

#include "Grid.h"

#ifdef PARALLEL_GRID

#include <mpi.h>

/**
 * Base grid of parallel grid and parallel grid coordinate
 */
#ifdef GRID_1D
#define ParallelGridBase Grid<GridCoordinate1D>
#define ParallelGridCoordinateTemplate GridCoordinate1DTemplate
#define ParallelGridCoordinate GridCoordinate1D
#define ParallelGridCoordinateFP GridCoordinateFP1D
#endif /* GRID_1D */

#ifdef GRID_2D
#define ParallelGridBase Grid<GridCoordinate2D>
#define ParallelGridCoordinateTemplate GridCoordinate2DTemplate
#define ParallelGridCoordinate GridCoordinate2D
#define ParallelGridCoordinateFP GridCoordinateFP2D
#endif /* GRID_2D */

#ifdef GRID_3D
#define ParallelGridBase Grid<GridCoordinate3D>
#define ParallelGridCoordinateTemplate GridCoordinate3DTemplate
#define ParallelGridCoordinate GridCoordinate3D
#define ParallelGridCoordinateFP GridCoordinateFP3D
#endif /* GRID_3D */

/**
 * Parallel grid buffer types.
 */
enum BufferPosition
{
#define FUNCTION(X) X,
#include "BufferPosition.inc.h"
}; /* BufferPosition */

/**
 * Class with data shared between all parallel grids on a single computational node
 */
class ParallelGridCore
{
private:

  /**
   * Opposite buffer position corresponding to buffer position
   */
  std::vector<BufferPosition> oppositeDirections;

  /**
   * Current node (process) identificator.
   */
  int processId;

  /**
   * Overall count of nodes (processes).
   */
  int totalProcCount;

  /**
   * Process ids corresponding to directions
   */
  std::vector<int> directions;

  /**
   * Flags corresponding to direction, whether send and receive procedures should be performed for this direction
   */
  std::vector< std::pair<bool, bool> > doShare;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)

  /**
   * Number of nodes in the grid by Ox axis
   */
  int nodeGridSizeX;

#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)

  /**
   * Number of nodes in the grid by Oy axis
   */
  int nodeGridSizeY;

#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D

  /**
   * Number of nodes in the grid by Oz axis
   */
  int nodeGridSizeZ;

#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeY
   */
  int nodeGridSizeXY;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ

  /**
   * Precalculated value of nodeGridSizeY * nodeGridSizeZ
   */
  int nodeGridSizeYZ;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeZ
   */
  int nodeGridSizeXZ;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   */
  int nodeGridSizeXYZ;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has left neighbour
   */
  bool hasL;

  /**
   * Flag whether computational node has right neighbour
   */
  bool hasR;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has down neighbour
   */
  bool hasD;

  /**
   * Flag whether computational node has up neighbour
   */
  bool hasU;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has back neighbour
   */
  bool hasB;

  /**
   * Flag whether computational node has front neighbour
   */
  bool hasF;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef DYNAMIC_GRID
  /**
   * Clock counter for calculations of current process
   */
  timespec calcClock;

  /**
   * Clock counter for share operations of current process
   */
  timespec shareClock;

  /**
   * Clock counter for start of calculations of current process
   */
  timespec calcStart;

  /**
   * Clock counter for stop of calculations of current process
   */
  timespec calcStop;

  /**
   * Clock counter for start of share operations of current process
   */
  timespec shareStart;

  /**
   * Clock counter for stop of share operations of current process
   */
  timespec shareStop;

  /**
   * Clock counters for calculations for all processes
   */
  std::vector<timespec> calcClockAll;

  /**
   * Clock counters for share operations for all processes
   */
  std::vector<timespec> shareClockAll;
#endif /* DYNAMIC_GRID */

  /**
   * Flag whether to use manual virtual topology or not
   */
  bool doUseManualTopology;

  /**
   * Virtual topology size, which was specified manually (not used in case of optimal virtual topology)
   */
  ParallelGridCoordinate topologySize;

  /**
   * Communicator for all processes, used in computations
   * (could differ from MPI_COMM_WORLD on the processes, which are not used in computations)
   */
  MPI_Comm communicator;

private:

  void initOppositeDirections ();
  BufferPosition getOpposite (BufferPosition);

  void getShare (BufferPosition, std::pair<bool, bool> &);

  /*
   * TODO: make names start with lower case
   */
  void NodeGridInit (ParallelGridCoordinate);
  void ParallelGridCoreConstructor (ParallelGridCoordinate);
  void InitBufferFlags ();
  void InitDirections ();

#ifdef DYNAMIC_GRID
  void timespec_diff (struct timespec *, struct timespec *, struct timespec *);
#endif /* DYNAMIC_GRID */

public:

  ParallelGridCore (int, int, ParallelGridCoordinate, bool, ParallelGridCoordinate);

  /**
   * Getter for communicator for all processes, used in computations
   *
   * @return communicator for all processes, used in computations
   */
  MPI_Comm getCommunicator () const
  {
    return communicator;
  } /* getCommunicator */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)

  /**
   * Getter for number of nodes in the grid by Ox axis
   *
   * @return number of nodes in the grid by Ox axis
   */
  int getNodeGridSizeX () const
  {
    return nodeGridSizeX;
  } /* getNodeGridSizeX */

#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)

  /**
   * Getter for number of nodes in the grid by Oy axis
   *
   * @return number of nodes in the grid by Oy axis
   */
  int getNodeGridSizeY () const
  {
    return nodeGridSizeY;
  } /* getNodeGridSizeY */

#endif /* GRID_2D || GRID_3D */

#ifdef GRID_3D

  /**
   * Getter for number of nodes in the grid by Oz axis
   *
   * @return number of nodes in the grid by Oz axis
   */
  int getNodeGridSizeZ () const
  {
    return nodeGridSizeZ;
  } /* getNodeGridSizeZ */

#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeY
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeY
   */
  int getNodeGridSizeXY () const
  {
    return nodeGridSizeXY;
  } /* getNodeGridSizeXY */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ

  /**
   * Getter for precalculated value of nodeGridSizeY * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeY * nodeGridSizeZ
   */
  int getNodeGridSizeYZ () const
  {
    return nodeGridSizeYZ;
  } /* getNodeGridSizeYZ */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeZ
   */
  int getNodeGridSizeXZ () const
  {
    return nodeGridSizeXZ;
  } /* getNodeGridSizeXZ */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   */
  int getNodeGridSizeXYZ () const
  {
    return nodeGridSizeXYZ;
  } /* getNodeGridSizeXYZ */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Ox axis for current process
   *
   * @return coordinate of process in the nodes' grid by Ox axis for current process
   */
  int getNodeGridX () const
  {
    return getNodeGridX (processId);
  }
  int getNodeGridX (int) const;

  /**
   * Getter for flag whether computational node has left neighbour
   *
   * @return flag whether computational node has left neighbour
   */
  bool getHasL () const
  {
    return hasL;
  } /* getHasL */

  /**
   * Getter for flag whether computational node has right neighbour
   *
   * @return flag whether computational node has right neighbour
   */
  bool getHasR () const
  {
    return hasR;
  } /* getHasR */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Oy axis for current process
   *
   * @return coordinate of process in the nodes' grid by Oy axis for current process
   */
  int getNodeGridY () const
  {
    return getNodeGridY (processId);
  }
  int getNodeGridY (int) const;

  /**
   * Getter for flag whether computational node has down neighbour
   *
   * @return flag whether computational node has down neighbour
   */
  bool getHasD () const
  {
    return hasD;
  } /* getHasD */

  /**
   * Getter for flag whether computational node has up neighbour
   *
   * @return flag whether computational node has up neighbour
   */
  bool getHasU () const
  {
    return hasU;
  } /* getHasU */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Oz axis for current process
   *
   * @return coordinate of process in the nodes' grid by Oz axis for current process
   */
  int getNodeGridZ () const
  {
    return getNodeGridZ (processId);
  }
  int getNodeGridZ (int) const;

  /**
   * Getter for flag whether computational node has back neighbour
   *
   * @return flag whether computational node has back neighbour
   */
  bool getHasB () const
  {
    return hasB;
  } /* getHasB */

  /**
   * Getter for flag whether computational node has front neighbour
   *
   * @return flag whether computational node has front neighbour
   */
  bool getHasF () const
  {
    return hasF;
  } /* getHasF */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  /**
   * Getter for id of process corresponding to current computational node
   *
   * @return id of process corresponding to current computational node
   */
  int getProcessId () const
  {
    return processId;
  } /* getProcessId */

  /**
   * Getter for overall count of nodes (processes)
   *
   * @return overall count of nodes (processes)
   */
  int getTotalProcCount () const
  {
    return totalProcCount;
  } /* getTotalProcCount */

  /**
   * Getter for flags corresponding to direction, whether send and receive procedures should be performed
   * for this direction
   *
   * @return flags corresponding to direction, whether send and receive procedures should be performed
   * for this direction
   */
  const std::vector< std::pair<bool, bool> > &getDoShare ()
  {
    return doShare;
  } /* getDoShare */

  /**
   * Getter for opposite buffer position corresponding to buffer position
   *
   * @return opposite buffer position corresponding to buffer position
   */
  const std::vector<BufferPosition> &getOppositeDirections ()
  {
    return oppositeDirections;
  } /* getOppositeDirections */

  /**
   * Getter for process ids corresponding to directions
   *
   * @return process ids corresponding to directions
   */
  const std::vector<int> &getDirections ()
  {
    return directions;
  } /* getDirections */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  void initOptimal (grid_coord, grid_coord, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void initOptimal (grid_coord, grid_coord, grid_coord, int &, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef DYNAMIC_GRID

  void StartCalcClock ();
  void StopCalcClock ();

  void StartShareClock ();
  void StopShareClock ();

  void ShareClocks ();
  void ClearClocks ();

  /**
   * Getter for calculations clock
   *
   * @return calculations clock
   */
  timespec getCalcClock () const
  {
    return calcClock;
  } /* getCalcClock */

  /**
   * Getter for share clock
   *
   * @return share clock
   */
  timespec getShareClock () const
  {
    return shareClock;
  } /* getShareClock */

#endif /* DYNAMIC_GRID */

  /*
   * TODO: move out of ParallelGridCore
   */
  /**
   * Find greatest common divider of two integer numbers
   *
   * @return greatest common divider of two integer numbers
   */
  static grid_coord greatestCommonDivider (grid_coord a, /**< first integer number */
                                           grid_coord b) /**< second integer number */
  {
    if (b == 0)
    {
      return a;
    }
    else
    {
      return greatestCommonDivider (b, a % b);
    }
  } /* greatestCommonDivider */
}; /* ParallelGridCore */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_CORE_H */

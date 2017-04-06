#ifndef PARALLEL_GRID_CORE_H
#define PARALLEL_GRID_CORE_H

#include "Grid.h"

#ifdef PARALLEL_GRID

/**
 * Base grid of parallel grid and parallel grid coordinate
 */
#ifdef GRID_1D
#define ParallelGridBase Grid<GridCoordinate1D>
#define ParallelGridCoordinate GridCoordinate1D
#define ParallelGridCoordinateFP GridCoordinateFP1D
#endif /* GRID_1D */

#ifdef GRID_2D
#define ParallelGridBase Grid<GridCoordinate2D>
#define ParallelGridCoordinate GridCoordinate2D
#define ParallelGridCoordinateFP GridCoordinateFP2D
#endif /* GRID_2D */

#ifdef GRID_3D
#define ParallelGridBase Grid<GridCoordinate3D>
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

public:

  ParallelGridCore (int, int, ParallelGridCoordinate);

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

  int getNodeGridX () const;

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

  int getNodeGridY () const;

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

  int getNodeGridZ () const;

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

  void FindProportionForNodeGrid (int &, int &, int &, FPValue);
  void initOptimal (grid_coord, grid_coord, int &, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void FindProportionForNodeGrid (int &, int &, int &, int &, FPValue, FPValue);
  void initOptimal (grid_coord, grid_coord, grid_coord, int &, int &, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */


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

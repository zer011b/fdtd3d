#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>
#include <vector>

#include "FieldPoint.h"

typedef uint16_t grid_size;

/**
 * Size of the grid
 */
class GridSize
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_size sizeX;
#if defined (GRID_2D) || defined (GRID_3D)
  grid_size sizeY;
#if defined (GRID_3D)
  grid_size sizeZ;
#endif
#endif
#endif

public:

  // Constructor for all cases
  GridSize (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_size sx = 0
#if defined (GRID_2D) || defined (GRID_3D)
    , grid_size sy = 0
#if defined (GRID_3D)
    , grid_size sz = 0
#endif
#endif
#endif
  );

  ~GridSize ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_size getTotalSize ();
  grid_size getSizeX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_size getSizeY ();
#if defined (GRID_3D)
  grid_size getSizeZ ();
#endif
#endif
#endif
};

/**
 * Grid itself
 */
class Grid
{
  GridSize size;
  std::vector<FieldPointValue> gridValues;

public:

  Grid (GridSize s);
  ~Grid ();
};

#endif /* FIELD_GRID_H */
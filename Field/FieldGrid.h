#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>
#include <vector>

#include "FieldPoint.h"

typedef uint16_t grid_coord;

/**
 * Size of the grid
 */
class GridCoordinate
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord x;
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord y;
#if defined (GRID_3D)
  grid_coord z;
#endif
#endif
#endif

public:

  // Constructor for all cases
  GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord sx = 0
#if defined (GRID_2D) || defined (GRID_3D)
    , grid_coord sy = 0
#if defined (GRID_3D)
    , grid_coord sz = 0
#endif
#endif
#endif
  );

  ~GridCoordinate ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord getTotalCoord ();
  grid_coord getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord getY ();
#if defined (GRID_3D)
  grid_coord getZ ();
#endif
#endif
#endif
};


typedef std::vector<FieldPointValue> VectorFieldPointValues;
/**
 * Grid itself
 */
class Grid
{
  GridCoordinate size;
  VectorFieldPointValues gridValues;

private:

  bool isLegitIndex (GridCoordinate& position);
  grid_coord calculateIndex (GridCoordinate& position);

public:

  Grid (GridCoordinate& s);
  ~Grid ();

  GridCoordinate& getSize ();
  VectorFieldPointValues& getValues ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  void setFieldPointValue (FieldPointValue& value, GridCoordinate& position);
  FieldPointValue& getFieldPointValue (GridCoordinate& position);
#endif
};

#endif /* FIELD_GRID_H */
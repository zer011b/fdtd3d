#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>
#include <vector>

#include "FieldPoint.h"


// Type of one-dimensional coordinate.
typedef uint16_t grid_coord;
// Type of three-dimensional coordinate.
typedef uint64_t grid_iter;


// Coordinate in the grid.
class GridCoordinate
{
  // One dimensional coordinates.
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

  // Constructor for all cases.
  GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    const grid_coord& sx = 0
#if defined (GRID_2D) || defined (GRID_3D)
    , const grid_coord& sy = 0
#if defined (GRID_3D)
    , const grid_coord& sz = 0
#endif
#endif
#endif
  );

  ~GridCoordinate ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  // Calculate three-dimensional coordinate.
  grid_iter calculateTotalCoord () const;

  // Get one-dimensional coordinates.
  const grid_coord& getX () const;
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& getY () const;
#if defined (GRID_3D)
  const grid_coord& getZ () const;
#endif
#endif
#endif
};


// Vector of points in grid.
typedef std::vector<FieldPointValue> VectorFieldPointValues;


// Grid itself.
class Grid
{
  // Size of the grid.
  GridCoordinate size;

  // Vector of points in grid.
  VectorFieldPointValues gridValues;

private:

  // Check whether position is appropriate to get/set value from.
  bool isLegitIndex (const GridCoordinate& position) const;

  // Calculate three-dimensional coordinate from position.
  grid_iter calculateIndexFromPosition (const GridCoordinate& position) const;

public:

  Grid (const GridCoordinate& s);
  ~Grid ();

  // Get size of the grid.
  const GridCoordinate& getSize () const;

  // Get values in the grid.
  VectorFieldPointValues& getValues ();

  // Calculate position from three-dimensional coordinate.
  GridCoordinate calculatePositionFromIndex (grid_iter index) const;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  // Set field point at coordinate in grid.
  void setFieldPointValue (const FieldPointValue& value, const GridCoordinate& position);

  // Get field point at coordinate in grid.
  FieldPointValue& getFieldPointValue (const GridCoordinate& position);
#endif

  // Replace previous layer with current and so on.
  void shiftInTime ();
};


#endif /* FIELD_GRID_H */

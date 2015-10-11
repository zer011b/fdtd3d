#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>
#include <vector>

#include "FieldPoint.h"


typedef uint16_t grid_coord;
typedef uint64_t grid_iter;


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
  grid_coord calculateTotalCoord () const;
  const grid_coord& getX () const;
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& getY () const;
#if defined (GRID_3D)
  const grid_coord& getZ () const;
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

  bool isLegitIndex (const GridCoordinate& position) const;
  grid_coord calculateIndexFromPosition (const GridCoordinate& position) const;

public:

  Grid (const GridCoordinate& s);
  ~Grid ();

  const GridCoordinate& getSize () const;
  VectorFieldPointValues& getValues ();

  GridCoordinate calculatePositionFromIndex (grid_coord index) const;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  void setFieldPointValue (const FieldPointValue& value, const GridCoordinate& position);
  FieldPointValue& getFieldPointValue (const GridCoordinate& position);
#endif
};


#endif /* FIELD_GRID_H */
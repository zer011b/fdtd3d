#ifndef GRID_COORDINATE_H
#define GRID_COORDINATE_H

#include <stdint.h>

// Type of one-dimensional coordinate.
typedef uint32_t grid_coord;
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
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

public:

  // Constructor for all cases.
  GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    const grid_coord& sx
#if defined (GRID_2D) || defined (GRID_3D)
    , const grid_coord& sy
#if defined (GRID_3D)
    , const grid_coord& sz
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D*/
#endif /* GRID_1D || GRID_2D || GRID_3D*/
  );

#if defined (GRID_2D) || defined (GRID_3D)
  // Constructor for case when grid has the same dimension for all axes.
  GridCoordinate (const grid_coord& s);
#endif

  GridCoordinate ();
  GridCoordinate (const GridCoordinate& pos);

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
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D*/
#endif /* GRID_1D || GRID_2D || GRID_3D*/

  // Set one-dimensional coordinates.
  void setX (const grid_coord& new_x);
#if defined (GRID_2D) || defined (GRID_3D)
  void setY (const grid_coord& new_y);
#if defined (GRID_3D)
  void setZ (const grid_coord& new_z);
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D*/

  friend GridCoordinate operator+ (GridCoordinate lhs, const GridCoordinate& rhs)
  {
#if defined (GRID_1D)
    return GridCoordinate (lhs.getX () + rhs.getX ());
#endif /* GRID_1D */
#if defined (GRID_2D)
    return GridCoordinate (lhs.getX () + rhs.getX (), lhs.getY () + rhs.getY ());
#endif /* GRID_2D */
#if defined (GRID_3D)
    return GridCoordinate (lhs.getX () + rhs.getX (), lhs.getY () + rhs.getY (), lhs.getZ () + rhs.getZ ());
#endif /* GRID_3D */
  }

  bool operator== (const GridCoordinate& rhs)
  {
#if defined (GRID_1D)
    return getX () == rhs.getX ();
#endif /* GRID_1D */
#if defined (GRID_2D)
    return getX () == rhs.getX () && getY () == rhs.getY ();
#endif /* GRID_2D */
#if defined (GRID_3D)
    return getX () == rhs.getX () && getY () == rhs.getY () && getZ () == rhs.getZ ();
#endif /* GRID_3D */
  }
};

#endif /* GRID_COORDINATE_H */

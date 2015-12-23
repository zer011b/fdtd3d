#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>
#include <vector>

#include "FieldPoint.h"


enum BufferPosition
{
#if defined (GRID_1D)
  LEFT,
  RIGHT
#endif
#if defined (GRID_2D)
  LEFT,
  RIGHT,
  UP,
  DOWN,
  LEFT_UP,
  LEFT_DOWN,
  RIGHT_UP,
  RIGHT_DOWN
#endif
#if defined (GRID_3D)
  LEFT,
  RIGHT,
  UP,
  DOWN,
  FRONT,
  BACK,
  LEFT_FRONT,
  LEFT_BACK,
  LEFT_UP,
  LEFT_DOWN,
  RIGHT_FRONT,
  RIGHT_BACK,
  RIGHT_UP,
  RIGHT_DOWN,
  UP_FRONT,
  UP_BACK,
  DOWN_FRONT,
  DOWN_BACK,
  LEFT_UP_FRONT,
  LEFT_UP_BACK,
  LEFT_DOWN_FRONT,
  LEFT_DOWN_BACK
  RIGHT_UP_FRONT,
  RIGHT_UP_BACK,
  RIGHT_DOWN_FRONT,
  RIGHT_DOWN_BACK
#endif
};


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

  GridCoordinate (const GridCoordinate& pos)
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    x = pos.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
    y = pos.getY ();
#if defined (GRID_3D)
    z = pos.getZ ();
#endif
#endif
#endif
  }

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

  friend GridCoordinate operator+ (GridCoordinate lhs, const GridCoordinate& rhs)
  {
#if defined (GRID_1D)
    return GridCoordinate (lhs.getX () + rhs.getX ());
#endif
#if defined (GRID_2D)
    return GridCoordinate (lhs.getX () + rhs.getX (), lhs.getY () + rhs.getY ());
#endif
#if defined (GRID_3D)
    return GridCoordinate (lhs.getX () + rhs.getX (), lhs.getY () + rhs.getY (), lhs.getZ () + rhs.getZ ());
#endif
  }
};


// Vector of points in grid.
typedef std::vector<FieldPointValue*> VectorFieldPointValues;


// Grid itself.
class Grid
{
  // Size of the grid.
  GridCoordinate size;

  // Vector of points in grid.
  // Owns this. Deletes all FieldPointValue* itself.
  VectorFieldPointValues gridValues;

#if defined (PARALLEL_GRID)
  int processId;

  int totalProcCount;

  // Size of current piece
  GridCoordinate currentSize;

  // Size of buffer zone
  GridCoordinate bufferSizeLeft;
  GridCoordinate bufferSizeRight;

  // Size of current piece
  GridCoordinate totalSize;
#endif

private:

  // Check whether position is appropriate to get/set value from.
  bool isLegitIndex (const GridCoordinate& position) const;

  // Calculate three-dimensional coordinate from position.
  grid_iter calculateIndexFromPosition (const GridCoordinate& position) const;

  // Get values in the grid.
  VectorFieldPointValues& getValues ();

public:

#if defined (PARALLEL_GRID)
  Grid (const GridCoordinate& totSize, const GridCoordinate& curSize,
        const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
        const int process, const int totalProc);
#else
  Grid (const GridCoordinate& s);
#endif

  ~Grid ();

  // Get size of the grid.
  const GridCoordinate& getSize () const;


  // Calculate position from three-dimensional coordinate.
  GridCoordinate calculatePositionFromIndex (grid_iter index) const;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  // Set field point at coordinate in grid.
  void setFieldPointValue (FieldPointValue* value, const GridCoordinate& position);

  // Get field point at coordinate in grid.
  FieldPointValue* getFieldPointValue (const GridCoordinate& position);
  FieldPointValue* getFieldPointValue (grid_iter coord);
/*#if defined (PARALLEL_GRID)
  FieldPointValue* getFieldPointValueGlobal (const GridCoordinate& position);
  FieldPointValue* getFieldPointValueGlobal (grid_iter coord);
#endif*/
#endif

#if defined (PARALLEL_GRID)
  void SendBuffer (BufferPosition buffer, int processTo);
  void ReceiveBuffer (BufferPosition buffer, int processFrom);
#endif

  // Replace previous layer with current and so on.
  void shiftInTime ();
};


#endif /* FIELD_GRID_H */

#ifndef FIELD_GRID_H
#define FIELD_GRID_H

#include <cstdint>

typedef uint16_t grid_size;

class GridSize
{
private:

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
};

#endif /* FIELD_GRID_H */
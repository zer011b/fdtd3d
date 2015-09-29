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

  GridSize ();
  ~GridSize ();
};

#endif /* FIELD_GRID_H */
#include "FieldGrid.h"

GridSize::GridSize (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_size sx
#if defined (GRID_2D) || defined (GRID_3D)
  , grid_size sy
#if defined (GRID_3D)
  , grid_size sz
#endif
#endif
#endif
  ) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  sizeX (sx)
#if defined (GRID_2D) || defined (GRID_3D)
  , sizeY (sy)
#if defined (GRID_3D)
  , sizeZ (sz)
#endif
#endif
#endif 
{
}

GridSize::~GridSize ()
{
}
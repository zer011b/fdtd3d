#include <iostream>

#include "BMPLoader.h"

void
BMPLoader::LoadGrid (Grid& grid) const
{
  #if defined (GRID_1D)
    load1D (grid);
  #else
  #if defined (GRID_2D)
    load2D (grid);
  #else
  #if defined (GRID_3D)
    load3D (grid);
  #endif
  #endif
  #endif
}

#if defined (GRID_1D)
void
BMPLoader::load1D (Grid& grid) const
{

}
#endif

#if defined (GRID_2D)
void
BMPLoader::load2D (Grid& grid) const
{

}
#endif

#if defined (GRID_3D)
void
BMPLoader::load3D (Grid& grid) const
{

}
#endif

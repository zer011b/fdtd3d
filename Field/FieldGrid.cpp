#include <iostream>

#include "FieldGrid.h"

// ================================ GridSize ================================
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

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_size GridSize::getSizeX ()
{
  return sizeX;
}
#if defined (GRID_2D) || defined (GRID_3D)
grid_size GridSize::getSizeY ()
{
  return sizeY;
}
#if defined (GRID_3D)
grid_size GridSize::getSizeZ ()
{
  return sizeZ;
}
#endif
#endif
#endif

grid_size GridSize::getTotalSize ()
{
#if defined (GRID_1D)
  return sizeX;
#else
#if defined (GRID_2D)
  return sizeX * sizeY;
#else
#if defined (GRID_3D)
  return sizeX * sizeY * sizeZ;
#endif
#endif
#endif
}

// ================================ Grid ================================
Grid::Grid(GridSize s) :
  size (s)
{
  gridValues.resize (size.getTotalSize ());
  std::cout << "New grid with size: " << gridValues.size () << std::endl;
}

Grid::~Grid ()
{
}
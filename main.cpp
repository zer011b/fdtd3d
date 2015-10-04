#include <iostream>

#include "FieldGrid.h"

int main (int argc, char** argv)
{
  GridCoordinate size (10, 10, 10);
  Grid grid (size);

  FieldPointValue val (-1.5, 1200.0, 10.0);
  GridCoordinate pos (2, 4, 8);
  grid.setFieldPointValue(val, pos);

  std::cout << "Main." << std::endl;
  return 0;
}
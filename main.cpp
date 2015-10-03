#include <iostream>

#include "FieldGrid.h"

int main (int argc, char** argv)
{
  GridSize s (10, 10, 10);
  Grid grid (s);

  std::cout << "Main." << std::endl;
  return 0;
}
#include <iostream>

#include "FieldGrid.h"
#include "BMPDumper.h"

int main (int argc, char** argv)
{
  GridCoordinate size (3);
  Grid grid (size);

  FieldPointValue val1 (0, 100, 200);
  GridCoordinate pos1 (0);
  grid.setFieldPointValue(val1, pos1);

  FieldPointValue val2 (-50, -100, -150);
  GridCoordinate pos2 (1);
  grid.setFieldPointValue(val2, pos2);

  FieldPointValue val3 (25, 25, 25);
  GridCoordinate pos3 (2);
  grid.setFieldPointValue(val3, pos3);


  FieldPointValue& val_1 = grid.getFieldPointValue (pos2);
  std::cout << val_1.getCurValue () << ", " <<
    val_1.getPrevValue() << ", " << val_1.getPrevPrevValue() << std::endl;

  BMPDumper dumper;
  dumper.dumpGrid (grid);

  std::cout << "Main." << std::endl;
  return 0;
}
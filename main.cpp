#include <iostream>

#include "FieldGrid.h"
#include "BMPDumper.h"

int main (int argc, char** argv)
{
  GridCoordinate size (3, 3);
  Grid grid (size);

  FieldPointValue val1 (0, 100, 100);
  GridCoordinate pos1 (0, 0);
  grid.setFieldPointValue(val1, pos1);

  FieldPointValue val2 (0, 0, 0);
  GridCoordinate pos2 (1, 0);
  grid.setFieldPointValue(val2, pos2);

  FieldPointValue val3 (100, 25, 25);
  GridCoordinate pos3 (2, 0);
  grid.setFieldPointValue(val3, pos3);

  FieldPointValue val4 (0, 100, 100);
  GridCoordinate pos4 (0, 1);
  grid.setFieldPointValue(val4, pos4);

  FieldPointValue val5 (100, 100, 15);
  GridCoordinate pos5 (1, 1);
  grid.setFieldPointValue(val5, pos5);

  FieldPointValue val6 (100, 10, 100);
  GridCoordinate pos6 (2, 1);
  grid.setFieldPointValue(val6, pos6);

  FieldPointValue val7 (0, 75, 75);
  GridCoordinate pos7 (0, 2);
  grid.setFieldPointValue(val7, pos7);

  FieldPointValue val8 (0, 0, 0);
  GridCoordinate pos8 (1, 2);
  grid.setFieldPointValue(val8, pos8);

  FieldPointValue val9 (100, 0, 100);
  GridCoordinate pos9 (2, 2);
  grid.setFieldPointValue(val9, pos9);

  FieldPointValue& val_1 = grid.getFieldPointValue (pos1);
  std::cout << val_1.getCurValue () << ", " <<
    val_1.getPrevValue() << ", " << val_1.getPrevPrevValue() << std::endl;

  BMPDumper dumper;
  dumper.dumpGrid (grid);

  std::cout << "Main." << std::endl;
  return 0;
}
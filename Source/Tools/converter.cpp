#include <iostream>

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "Settings.h"
#include "GridInterface.h"

int main (int argc, char **argv)
{
  FileType fromType;
  FileType toType;

  grid_coord sizex = 0;
  grid_coord sizey = 0;
  grid_coord sizez = 0;

  int dim = 0;

  if (argc == 1)
  {
    printf ("example: ./converter --txt-to-bmp --file 1.txt)\n");
    return 1;
  }

  std::string input;

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp (argv[i], "--file") == 0)
    {
      ++i;
      input = std::string (argv[i]);
    }
    else if (strcmp (argv[i], "--txt-to-bmp") == 0)
    {
      fromType = FILE_TYPE_TXT;
      toType = FILE_TYPE_BMP;
    }
    else if (strcmp (argv[i], "--sizex") == 0)
    {
      ++i;
      sizex = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizey") == 0)
    {
      ++i;
      sizex = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizez") == 0)
    {
      ++i;
      sizex = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--dim") == 0)
    {
      ++i;
      dim = STOI (argv[i]);
      ALWAYS_ASSERT (dim > 0 && dim < 4);
    }
    else
    {
      return 1;
    }
  }

  std::string output;
  switch (toType)
  {
    case FILE_TYPE_TXT:
    {
      output = input + std::string (".txt");
      break;
    }
    case FILE_TYPE_DAT:
    {
      output = input + std::string (".dat");
      break;
    }
    case FILE_TYPE_BMP:
    {
      output = input + std::string (".bmp");
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (dim == 3)
  {
    GridCoordinate3D zero = GRID_COORDINATE_3D (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    GridCoordinate3D size = GRID_COORDINATE_3D (sizex, sizey, sizez, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    Grid<GridCoordinate3D> grid (size, 0, 1, "tmp_grid");

    std::vector< std::string > fileNames (1);
    fileNames[0] = input;

    TXTLoader<GridCoordinate3D> txtLoader3D;
    txtLoader3D.loadGrid (&grid, zero, grid.getSize (), 0, 0, fileNames);
  }
  else
  {
    // TODO: add other dimensions
    ALWAYS_ASSERT (0);
  }



  return 0;
}

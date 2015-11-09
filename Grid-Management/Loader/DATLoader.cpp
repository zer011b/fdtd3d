#include <iostream>
#include <fstream>

#include "DATLoader.h"


void
DATLoader::loadFromFile (Grid& grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::in | std::ios::binary);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::in | std::ios::binary);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::in | std::ios::binary);
      break;
    }
#endif
#endif
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (file.is_open());

  char* memblock = new char [sizeof (FieldValue)];

  // Go through all values and write to file.
  VectorFieldPointValues& values = grid.getValues ();
  grid_iter end = values.size ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = values[iter];
    ASSERT (current);

    switch (type)
    {
      case CURRENT:
      {
        file.read (memblock, sizeof (FieldValue));
        current->setCurValue (*((FieldValue*) memblock));
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        file.read (memblock, sizeof (FieldValue));
        current->setPrevValue (*((FieldValue*) memblock));
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        file.read (memblock, sizeof (FieldValue));
        current->setPrevPrevValue (*((FieldValue*) memblock));
        break;
      }
#endif
#endif
      default:
      {
        UNREACHABLE;
      }
    }
  }

  delete[] memblock;

  file.close();
}

void
DATLoader::loadGrid (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();
  std::cout << "Load grid from binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif
#endif

  std::cout << "Loaded. " << std::endl;
}

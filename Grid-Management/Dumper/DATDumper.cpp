#include <iostream>
#include <fstream>

#include "DATDumper.h"

void
DATDumper::writeToFile (Grid& grid, GridFileType type) const
{
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::out | std::ios::binary);
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
        file.write ((char*) &(current->getCurValue ()), sizeof (FieldValue));
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        file.write ((char*) &(current->getPrevValue ()), sizeof (FieldValue));
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        file.write ((char*) &(current->getPrevPrevValue ()), sizeof (FieldValue));
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

  file.close();
}

void
DATDumper::dumpGrid (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();
  std::cout << "Saving grid to binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  writeToFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    writeToFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    writeToFile (grid, PREVIOUS2);
  }
#endif
#endif

  std::cout << "Saved. " << std::endl;
}

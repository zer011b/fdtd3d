#ifndef DAT_DUMPER_H
#define DAT_DUMPER_H

#include <iostream>
#include <fstream>

#include "Dumper.h"

/**
 * Grid saver to binary files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class DATDumper: public Dumper<TCoord>
{
  // Save grid to file for specific layer.
  void writeToFile (Grid<TCoord> &grid, GridFileType type) const;

public:

  // Virtual method for grid saving.
  void dumpGrid (Grid<TCoord> &grid) const override;
};

/**
 * ======== Template implementation ========
 */

/**
 * Save grid to file for specific layer.
 */
template <class TCoord>
void
DATDumper<TCoord>::writeToFile (Grid<TCoord> &grid, GridFileType type) const
{
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = GridFileManager::cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = GridFileManager::prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = GridFileManager::prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (file.is_open());

  // Go through all values and write to file.
  grid_iter end = grid.getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    const FieldPointValue* current = grid.getFieldPointValue (iter);
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
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }
  }

  file.close();
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGrid (Grid<TCoord> &grid) const
{
  const TCoord& size = grid.getSize ();
  std::cout << "Saving grid to binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  writeToFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Saved. " << std::endl;
}

#endif /* DAT_DUMPER_H */

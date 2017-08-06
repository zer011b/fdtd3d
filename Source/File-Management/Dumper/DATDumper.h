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
  void writeToFile (Grid<TCoord> &grid, GridFileType type, TCoord, TCoord) const;

public:

  // Virtual method for grid saving.
  virtual void dumpGrid (Grid<TCoord> &grid, TCoord, TCoord) const CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Save grid to file for specific layer.
 */
template <class TCoord>
void
DATDumper<TCoord>::writeToFile (Grid<TCoord> &grid, GridFileType type, TCoord startCoord, TCoord endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
#ifdef CXX11_ENABLED
      std::string cur_dat = GridFileManager::cur + std::string (".dat");
#else
      std::string cur_dat = this->GridFileManager::cur + std::string (".dat");
#endif
      file.open (cur_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef CXX11_ENABLED
      std::string prev_dat = GridFileManager::prev + std::string (".dat");
#else
      std::string prev_dat = this->GridFileManager::prev + std::string (".dat");
#endif
      file.open (prev_dat.c_str (), std::ios::out | std::ios::binary);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef CXX11_ENABLED
      std::string prevPrev_dat = GridFileManager::prevPrev + std::string (".dat");
#else
      std::string prevPrev_dat = this->GridFileManager::prevPrev + std::string (".dat");
#endif
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
DATDumper<TCoord>::dumpGrid (Grid<TCoord> &grid, TCoord startCoord, TCoord endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  const TCoord& size = grid.getSize ();
  std::cout << "Saving grid to binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  writeToFile (grid, CURRENT, startCoord, endCoord);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    writeToFile (grid, PREVIOUS, startCoord, endCoord);
  }
#if defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    writeToFile (grid, PREVIOUS2, startCoord, endCoord);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Saved. " << std::endl;
}

#endif /* DAT_DUMPER_H */

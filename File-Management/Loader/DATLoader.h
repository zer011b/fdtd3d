#ifndef DAT_LOADER_H
#define DAT_LOADER_H

#include <iostream>
#include <fstream>

#include "Loader.h"

/**
 * Grid loader from binary files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class DATLoader: public Loader<TCoord>
{
  // Load grid from file for specific layer.
  void loadFromFile (Grid<TCoord> &grid, GridFileType type) const;

public:

  // Virtual method for grid loading.
#ifdef CXX11_ENABLED
  virtual void loadGrid (Grid<TCoord> &grid) const override;
#else
  virtual void loadGrid (Grid<TCoord> &grid) const;
#endif
};

/**
 * ======== Template implementation ========
 */

/**
 * Load grid from file for specific layer.
 */
template <class TCoord>
void
DATLoader<TCoord>::loadFromFile (Grid<TCoord> &grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = GridFileManager::cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::in | std::ios::binary);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = GridFileManager::prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::in | std::ios::binary);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = GridFileManager::prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::in | std::ios::binary);
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

  char* memblock = new char [sizeof (FieldValue)];

  // Go through all values and write to file.
  grid_iter end = grid.getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = grid.getFieldPointValue (iter);
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
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }
  }

  delete[] memblock;

  file.close();
}

/**
 * Virtual method for grid loading.
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGrid (Grid<TCoord> &grid) const
{
  const TCoord& size = grid.getSize ();
  std::cout << "Load grid from binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Loaded. " << std::endl;
}

#endif /* DAT_LOADER_H */

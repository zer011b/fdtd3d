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
  void loadFromFile (Grid<TCoord> *grid, TCoord, TCoord, int);

  void loadGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);

public:

  virtual ~DATLoader () {}

  // Virtual method for grid loading.
  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Load grid from file for specific layer.
 */
template <class TCoord>
void
DATLoader<TCoord>::loadFromFile (Grid<TCoord> *grid,
                                 TCoord startCoord,
                                 TCoord endCoord,
                                 int time_step_back)
{
  ASSERT ((time_step_back == -1) || (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ()));
#ifdef DEBUG_INFO
  TCoord zero = startCoord - startCoord;
  ASSERT (startCoord >= zero && startCoord < grid->getSize ());
  ASSERT (endCoord > zero && endCoord <= grid->getSize ());
#endif /* DEBUG_INFO */

  std::ifstream file;
  file.open (this->GridFileManager::names[time_step_back == -1 ? 0 : time_step_back].c_str (), std::ios::in | std::ios::binary);
  ASSERT (file.is_open());

  char memblock[sizeof (FieldValue)];

  // Go through all values and load from file.
  grid_coord end = grid->getSize().calculateTotalCoord ();
  for (grid_coord iter = 0; iter < end; ++iter)
  {
    TCoord pos = grid->calculatePositionFromIndex (iter);
    if (!(pos >= startCoord && pos < endCoord))
    {
      continue;
    }

    if (time_step_back == -1)
    {
      for (int i = 0; i < grid->getCountStoredSteps (); ++i)
      {
        file.read (memblock, sizeof (FieldValue));
        ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

        grid->setFieldValue (*((FieldValue*) memblock), iter, i);
      }
    }
    else
    {
      file.read (memblock, sizeof (FieldValue));
      ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

      grid->setFieldValue (*((FieldValue*) memblock), iter, time_step_back);
    }
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

template <class TCoord>
void
DATLoader<TCoord>::loadGridInternal (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                                     time_step timeStep, int time_step_back)
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Loading grid '" << grid->getName () << "' from binary. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Loading grid '" << grid->getName () << "' from binary. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  loadFromFile (grid, startCoord, endCoord, time_step_back);

  std::cout << "Loaded. " << std::endl;
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back, int pid)
{
  GridFileManager::setFileNames (time_step_back, timeStep, pid, std::string (grid->getName ()), FILE_TYPE_DAT);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back,
                             const std::vector< std::string > & customNames)
{
  GridFileManager::setCustomFileNames (customNames);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

#endif /* DAT_LOADER_H */

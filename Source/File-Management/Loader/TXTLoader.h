#ifndef TXT_LOADER_H
#define TXT_LOADER_H

#include <iostream>
#include <fstream>

#include "Loader.h"

/**
 * Grid loader from txt files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class TXTLoader: public Loader<TCoord>
{
  // Load grid from file for specific layer.
  void loadFromFile (Grid<TCoord> *grid, TCoord, TCoord, int);

  void loadGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);

public:

  virtual ~TXTLoader () {}

  // Virtual method for grid loading.
  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int) CXX11_OVERRIDE;
  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

template <class TCoord>
void
TXTLoader<TCoord>::loadGridInternal (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                                     time_step timeStep, int time_step_back)
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Loading grid '" << grid->getName () << "' from text. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Loading grid '" << grid->getName () << "' from text. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  if (time_step_back == -1)
  {
    /**
     * Save all time steps
     */
    for (int i = 0; i < grid->getCountStoredSteps (); ++i)
    {
      loadFromFile (grid, startCoord, endCoord, i);
    }
  }
  else
  {
    loadFromFile (grid, startCoord, endCoord, time_step_back);
  }

  std::cout << "Loaded. " << std::endl;
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back)
{
  int pid = 0;

#ifdef PARALLEL_GRID
  if (SOLVER_SETTINGS.getDoUseParallelGrid ())
  {
    pid = ParallelGrid::getParallelCore ()->getProcessId ();
  }
#endif /* PARALLEL_GRID */

  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_TXT);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back,
                             const std::vector< std::string > & customNames)
{
  GridFileManager::setCustomFileNames (customNames);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

#endif /* TXT_LOADER_H */

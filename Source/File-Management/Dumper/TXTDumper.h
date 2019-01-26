#ifndef TXT_DUMPER_H
#define TXT_DUMPER_H

#include <iostream>
#include <fstream>

#include "Dumper.h"

/**
 * Grid saver to txt files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class TXTDumper: public Dumper<TCoord>
{
  // Save grid to file for specific layer.
  void writeToFile (Grid<TCoord> *grid, TCoord, TCoord, int);

  void dumpGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);

public:

  virtual ~TXTDumper () {}

  // Virtual method for grid saving.
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

template <class TCoord>
void
TXTDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                                     time_step timeStep, int time_step_back)
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Saving grid '" << grid->getName () << "' to text. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Saving grid '" << grid->getName () << "' to text. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  if (time_step_back == -1)
  {
    /**
     * Save all time steps
     */
    for (int i = 0; i < grid->getCountStoredSteps (); ++i)
    {
      writeToFile (grid, startCoord, endCoord, i);
    }
  }
  else
  {
    writeToFile (grid, startCoord, endCoord, time_step_back);
  }

  std::cout << "Saved. " << std::endl;
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back, int pid)
{
  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_TXT);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back,
                             const std::vector< std::string > & customNames)
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

#endif /* TXT_DUMPER_H */

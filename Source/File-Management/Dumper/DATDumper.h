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
  void writeToFile (Grid<TCoord> *grid, TCoord, TCoord, int);

  void dumpGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);

public:

  virtual ~DATDumper () {}

  // Virtual method for grid saving.
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Save grid to file for specific layer.
 */
template <class TCoord>
void
DATDumper<TCoord>::writeToFile (Grid<TCoord> *grid,
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

  std::ofstream file;
  file.open (GridFileManager::names[time_step_back == -1 ? 0 : time_step_back].c_str (), std::ios::out | std::ios::binary);
  ASSERT (file.is_open());

  // Go through all values and write to file.
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
        file.write ((char*) (grid->getFieldValue (iter, i)), sizeof (FieldValue));
      }
    }
    else
    {
      file.write ((char*) (grid->getFieldValue (iter, time_step_back)), sizeof (FieldValue));
    }
  }

  file.close();
}

template <class TCoord>
void
DATDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                                     time_step timeStep, int time_step_back)
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Saving grid '" << grid->getName () << "' to binary. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Saving grid '" << grid->getName () << "' to binary. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  writeToFile (grid, startCoord, endCoord, time_step_back);

  std::cout << "Saved. " << std::endl;
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back)
{
  int pid = 0;

#ifdef PARALLEL_GRID
  if (SOLVER_SETTINGS.getDoUseParallelGrid ())
  {
    pid = ParallelGrid::getParallelCore ()->getProcessId ();
  }
#endif /* PARALLEL_GRID */

  GridFileManager::setFileNames (time_step_back, timeStep, pid, std::string (grid->getName ()), FILE_TYPE_DAT);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back,
                             const std::vector< std::string > & customNames)
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

#endif /* DAT_DUMPER_H */

#ifndef DAT_LOADER_H
#define DAT_LOADER_H

#include <iostream>
#include <fstream>

#include "Loader.h"

/**
 * Grid loader from binary files.
 */
template <class TCoord>
class DATLoader: public Loader<TCoord>
{
  void loadFromFile (Grid<TCoord> *, TCoord, TCoord, int);
  void loadGridInternal (Grid<TCoord> *, TCoord, TCoord, time_step, int);

public:

  virtual ~DATLoader () {}

  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
}; /* DATLoader */

/**
 * Load data from file
 */
template <class TCoord>
void
DATLoader<TCoord>::loadFromFile (Grid<TCoord> *grid, /**< grid to load */
                                 TCoord startCoord, /**< start loading from this coordinate */
                                 TCoord endCoord, /**< end loading at this coordinate */
                                 int time_step_back) /**< relative time step at which to load */
{
  ASSERT ((time_step_back == -1) || (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ()));
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > endCoord.getZero () && endCoord <= grid->getSize ());

  std::ifstream file;
  file.open (this->GridFileManager::names[time_step_back == -1 ? 0 : time_step_back].c_str (), std::ios::in | std::ios::binary);
  ASSERT (file.is_open());

  char memblock[sizeof (FieldValue)];

  // Go through all values and load from file.
  typename VectorFieldValues<TCoord>::Iterator iter (startCoord, startCoord, endCoord);
  typename VectorFieldValues<TCoord>::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (startCoord, endCoord);
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();

    if (time_step_back == -1)
    {
      for (int i = 0; i < grid->getCountStoredSteps (); ++i)
      {
        file.read (memblock, sizeof (FieldValue));
        ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

        grid->setFieldValue (*((FieldValue*) memblock), pos, i);
      }
    }
    else
    {
      file.read (memblock, sizeof (FieldValue));
      ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

      grid->setFieldValue (*((FieldValue*) memblock), pos, time_step_back);
    }
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
} /* DATLoader::loadFromFile */

/**
 * Choose scenario of loading of grid
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGridInternal (Grid<TCoord> *grid, /**< grid to load */
                                     TCoord startCoord, /**< start loading from this coordinate */
                                     TCoord endCoord, /**< end loading at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to load */
                                     int time_step_back) /**< relative time step at which to load */
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
} /* DATLoader::loadGridInternal */

/**
 * Virtual method for grid loading, which makes file names automatically
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             int pid) /**< pid of process, which does loading */
{
  GridFileManager::setFileNames (time_step_back, timeStep, pid, std::string (grid->getName ()), FILE_TYPE_DAT);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* DATLoader::loadGrid */

/**
 * Virtual method for grid loading, which uses custom names
 */
template <class TCoord>
void
DATLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* DATLoader::loadGrid */

#endif /* DAT_LOADER_H */

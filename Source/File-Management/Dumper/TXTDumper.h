#ifndef TXT_DUMPER_H
#define TXT_DUMPER_H

#include <iostream>
#include <fstream>

#include "Dumper.h"

/**
 * Grid saver to txt files.
 */
template <class TCoord>
class TXTDumper: public Dumper<TCoord>
{
  void writeToFile (Grid<TCoord> *, TCoord, TCoord, int);
  void dumpGridInternal (Grid<TCoord> *, TCoord, TCoord, time_step, int);
  void printLine (std::ofstream &, const TCoord &);

public:

  virtual ~TXTDumper () {}

  virtual void dumpGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
}; /* TXTDumper */

/**
 * Write data to file
 */
template <class TCoord>
void
TXTDumper<TCoord>::writeToFile (Grid<TCoord> *grid, /**< grid to save */
                                TCoord startCoord, /**< start saving from this coordinate */
                                TCoord endCoord, /**< end saving at this coordinate */
                                int time_step_back) /**< relative time step at which to save */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > endCoord.getZero () && endCoord <= grid->getSize ());

  std::ofstream file;
  file.open (this->GridFileManager::names[time_step_back].c_str (), std::ios::out);
  ASSERT (file.is_open());

  file << std::setprecision (std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  typename VectorFieldValues<TCoord>::Iterator iter (startCoord, startCoord, endCoord);
  typename VectorFieldValues<TCoord>::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (startCoord, endCoord);
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();

    printLine (file, pos);

    FieldValue *val = grid->getFieldValue (pos, time_step_back);

#ifdef COMPLEX_FIELD_VALUES
    file << val->real () << " " << val->imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
    file << *val << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
  }

  file.close();
} /* TXTDumper::writeToFile */

/**
 * Choose scenario of saving of grid
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, /**< grid to save */
                                     TCoord startCoord, /**< start saving from this coordinate */
                                     TCoord endCoord, /**< end saving at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to save */
                                     int time_step_back) /**< relative time step at which to save */
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
} /* TXTDumper::writeToFile */

/**
 * Virtual method for grid saving, which makes file names automatically
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             int pid) /**< pid of process, which does saving */
{
  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_TXT);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* TXTDumper::dumpGrid */

/**
 * Virtual method for grid saving, which uses custom names
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* TXTDumper::dumpGrid */

#endif /* TXT_DUMPER_H */

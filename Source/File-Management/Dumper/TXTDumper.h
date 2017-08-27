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
  void writeToFile (Grid<TCoord> *grid, GridFileType type, TCoord, TCoord) const;

public:

  // Virtual method for grid saving.
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord) const CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  const TCoord& size = grid->getSize ();
  std::cout << "Saving grid '" << grid->getName () << "' to text. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  writeToFile (grid, CURRENT, startCoord, endCoord);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS, startCoord, endCoord);
  }
#if defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS2, startCoord, endCoord);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Saved. " << std::endl;
}

#endif /* TXT_DUMPER_H */

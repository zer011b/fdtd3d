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
  void writeToFile (Grid<TCoord> &grid, GridFileType type) const;

public:

  // Virtual method for grid saving.
  void dumpGrid (Grid<TCoord> &grid) const CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
TXTDumper<TCoord>::dumpGrid (Grid<TCoord> &grid) const
{
  const TCoord& size = grid.getSize ();
  std::cout << "Saving grid to text. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  writeToFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    writeToFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    writeToFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Saved. " << std::endl;
}

#endif /* TXT_DUMPER_H */

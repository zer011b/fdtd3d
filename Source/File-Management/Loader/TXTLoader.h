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
  void loadFromFile (Grid<TCoord> *grid, GridFileType type) const;

public:

  // Virtual method for grid loading.
  virtual void loadGrid (Grid<TCoord> *grid) const CXX11_OVERRIDE;
};

/**
 * ======== Template implementation ========
 */

/**
 * Virtual method for grid loading.
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGrid (Grid<TCoord> *grid) const
{
  const TCoord& size = grid->getSize ();
  std::cout << "Load grid from binary. Size: " << size.calculateTotalCoord () << ". " << std::endl;

  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  std::cout << "Loaded. " << std::endl;
}

#endif /* TXT_LOADER_H */

#ifndef LOADER_H
#define LOADER_H

#include "Commons.h"

/**
 * Base class for all loaders.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class Loader: public GridFileManager
{
protected:

  // Protected constructor to disallow instantiation.
  Loader () {}

public:

  virtual ~Loader () {}

  // Pure virtual method for grid loading.
  virtual void loadGrid (Grid<TCoord> &grid) const = 0;
};

#endif /* LOADER_H */

#ifndef LOADER_H
#define LOADER_H

#include "Commons.h"

// Basic class for all loaders.
template <class TCoord>
class Loader: public GridFileManager
{
protected:

  Loader () {}

public:

  virtual void loadGrid (Grid<TCoord> &grid) const = 0;
  virtual ~Loader () {}
};

#endif /* LOADER_H */

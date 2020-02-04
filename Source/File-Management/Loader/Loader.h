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
public:

  virtual ~Loader () {}

  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) = 0;
  virtual void loadGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, const std::vector< std::string > &) = 0;
}; /* Loader */

#endif /* LOADER_H */

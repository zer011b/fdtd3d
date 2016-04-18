#ifndef DUMPER_H
#define DUMPER_H

#include "Commons.h"

/**
 * Base class for all dumpers.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class Dumper: public GridFileManager
{
protected:

  // Protected constructor to disallow instantiation.
  Dumper () {}

public:

  virtual ~Dumper () {}

  // Pure virtual method for grid saving.
  virtual void dumpGrid (Grid<TCoord> &grid) const = 0;
};

#endif /* DUMPER_H */

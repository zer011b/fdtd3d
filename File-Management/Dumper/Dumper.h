#ifndef DUMPER_H
#define DUMPER_H

#include "Commons.h"

// Basic class for all dumpers.
template <class TCoord>
class Dumper: public GridFileManager
{
protected:

  Dumper () {}

public:

  virtual void dumpGrid (Grid<TCoord> &grid) const = 0;
  virtual ~Dumper () {}
};

#endif /* DUMPER_H */

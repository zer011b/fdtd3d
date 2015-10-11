#ifndef DUMPER_H
#define DUMPER_H

#include "FieldGrid.h"

class Dumper
{
public:
  virtual void dumpGrid (Grid& grid, const grid_iter& timeStep) const = 0;
};

#endif /* DUMPER_H */
#ifndef DUMPER_H
#define DUMPER_H

#include "FieldGrid.h"

class Dumper
{
public:
  virtual void dumpGrid (Grid& grid) = 0;
};

#endif /* DUMPER_H */
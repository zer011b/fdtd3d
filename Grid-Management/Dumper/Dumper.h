#ifndef DUMPER_H
#define DUMPER_H

#include "Commons.h"

// Basic class for all dumpers.
class Dumper: public GridFileManager
{
public:

  virtual void dumpGrid (Grid& grid) const = 0;
};

#endif /* DUMPER_H */

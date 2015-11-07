#ifndef DUMPER_H
#define DUMPER_H

#include "Commons.h"

class Dumper: public GridFileManager
{
public:

  virtual void dumpGrid (Grid& grid) const = 0;
};

#endif /* DUMPER_H */

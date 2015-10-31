#ifndef DUMPER_H
#define DUMPER_H

#include "commons.h"

class Dumper: public GridFileManager
{
public:

  virtual void dumpGrid (Grid& grid) const = 0;
};

#endif /* DUMPER_H */

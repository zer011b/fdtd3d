#ifndef LOADER_H
#define LOADER_H

#include "Commons.h"

// Basic class for all loaders.
class Loader: public GridFileManager
{
public:

  virtual void loadGrid (Grid& grid) const = 0;
};

#endif /* LOADER_H */

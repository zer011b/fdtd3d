#ifndef LOADER_H
#define LOADER_H

#include "commons.h"

class Loader: public GridFileManager
{
public:

  virtual void LoadGrid (Grid& grid) const = 0;
};

#endif /* LOADER_H */

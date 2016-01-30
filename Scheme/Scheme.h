#ifndef SCHEME_H
#define SCHEME_H

#include "Grid.h"

class Scheme
{
public:
  virtual bool performStep () = 0;
};

#endif /* SCHEME_H */

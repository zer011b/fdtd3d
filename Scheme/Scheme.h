#ifndef SCHEME_H
#define SCHEME_H

#include "FieldGrid.h"

class Scheme
{
public:
  virtual bool performStep () = 0;
};

#endif /* SCHEME_H */

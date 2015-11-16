#ifndef SOLVER_H
#define SOLVER_H

#include "Scheme.h"

class Solver
{
  Scheme* scheme;

public:

  bool performStep ();

  Solver (Scheme* initScheme) :
    scheme (initScheme)
  {
    ASSERT (scheme);
  }

  ~Solver ()
  {
  }
};

#endif /* SOLVER_H */

#ifndef SOLVER_H
#define SOLVER_H

#include "Scheme.h"
#include "Assert.h"

class Solver
{
  Scheme* scheme;

public:

  void performSteps ();

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

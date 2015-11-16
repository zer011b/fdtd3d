#include "Solver.h"

bool
Solver::performStep ()
{
  return scheme->performStep ();
}

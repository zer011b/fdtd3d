#include "Solver.h"

void
Solver::performStep ()
{
  scheme->performStep ();
}

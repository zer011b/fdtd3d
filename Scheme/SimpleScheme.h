#ifndef SIMPLE_SCHEME_H
#define SIMPLE_SCHEME_H

#include "Scheme.h"

/**
 * Simple FDTD case: vacuum
 */
class SimpleScheme: public Scheme
{
  // Does not own this.
  Grid* E;
  Grid* H;
  Grid* Eps;
  Grid* Mu;

public:
  bool performStep () override
  {
    return true;
  }

  bool performStep1D ()
  {
    // Calculate E
    // for (grid_coord i = 1; i < E->getSize ().getX (); ++i)
    // {
    //   FieldValue cur = E->getFieldPointValue (i)->getCurValue ();
    //
    //   const FieldPointValue* H = H.getFieldPointValue (i);
    //
    //   FieldPointValue* value = new FieldPointValue ();
    // }
  }

  SimpleScheme (Grid* e, Grid* h, Grid* eps, Grid* mu);

  ~SimpleScheme ();
};

#endif /* SIMPLE_SCHEME_H */

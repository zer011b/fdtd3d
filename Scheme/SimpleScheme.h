#ifndef SIMPLE_SCHEME_H
#define SIMPLE_SCHEME_H

#include "Scheme.h"
#include "Grid.h"

/**
 * Simple FDTD case: vacuum
 */
class SimpleScheme: public Scheme
{
  // Does not own this.
  Grid<GridCoordinate1D> *E;
  Grid<GridCoordinate1D> *H;
  Grid<GridCoordinate1D> *Eps;
  Grid<GridCoordinate1D> *Mu;

public:
  void performStep () override
  {
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

  SimpleScheme (Grid<GridCoordinate1D> *e, Grid<GridCoordinate1D> *h, Grid<GridCoordinate1D> *eps, Grid<GridCoordinate1D> *mu);

  ~SimpleScheme ();
};

#endif /* SIMPLE_SCHEME_H */

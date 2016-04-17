#include "SimpleScheme.h"

SimpleScheme::SimpleScheme (Grid<GridCoordinate1D> *e, Grid<GridCoordinate1D> *h, Grid<GridCoordinate1D> *eps, Grid<GridCoordinate1D> *mu)
: E (e), H (h), Eps (eps), Mu (mu)
{
  // Verify.
  ASSERT (e);
  ASSERT (h);
  ASSERT (eps);
  ASSERT (mu);

  GridCoordinate1D sizeE = E->getSize ();
  GridCoordinate1D sizeH = H->getSize ();
  GridCoordinate1D sizeEps = Eps->getSize ();
  GridCoordinate1D sizeMu = Mu->getSize ();

#if defined (GRID_1D)
  ASSERT (sizeE.getX () == sizeH.getX ());
  ASSERT (sizeE.getX () == sizeEps.getX ());
  ASSERT (sizeE.getX () == sizeMu.getX ());
#if defined (GRID_2D)
  ASSERT (sizeE.getY () == sizeH.getY ());
  ASSERT (sizeE.getY () == sizeEps.getY ());
  ASSERT (sizeE.getY () == sizeMu.getY ());
#if defined (GRID_3D)
  ASSERT (sizeE.getZ () == sizeH.getZ ());
  ASSERT (sizeE.getZ () == sizeEps.getZ ());
  ASSERT (sizeE.getZ () == sizeMu.getZ ());
#endif
#endif
#endif
}

SimpleScheme::~SimpleScheme ()
{
}

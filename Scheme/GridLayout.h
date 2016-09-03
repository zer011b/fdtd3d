#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"
#include "Assert.h"

enum class LayoutDirection
{
  LEFT,
  RIGHT,
  DOWN,
  UP,
  BACK,
  FRONT
};

class GridLayout
{
public:

  GridLayout () {}
  virtual ~GridLayout () {}

  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;

  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;

  virtual GridCoordinate3D getExSize () const = 0;
  virtual GridCoordinate3D getEySize () const = 0;
  virtual GridCoordinate3D getEzSize () const = 0;

  virtual GridCoordinate3D getHxSize () const = 0;
  virtual GridCoordinate3D getHySize () const = 0;
  virtual GridCoordinate3D getHzSize () const = 0;

  virtual GridCoordinate3D getExStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getExEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEyStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEyEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEzStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEzEnd (GridCoordinate3D) const = 0;

  virtual GridCoordinate3D getHxStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHxEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHyStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHyEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHzStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHzEnd (GridCoordinate3D) const = 0;
};

class YeeGridLayout: public GridLayout
{
  const GridCoordinateFP3D zeroCoordFP;
  const GridCoordinate3D zeroCoord;

  const GridCoordinateFP3D minExCoordFP;
  const GridCoordinate3D minExCoord;
  GridCoordinate3D sizeEx;

  const GridCoordinateFP3D minEyCoordFP;
  const GridCoordinate3D minEyCoord;
  GridCoordinate3D sizeEy;

  const GridCoordinateFP3D minEzCoordFP;
  const GridCoordinate3D minEzCoord;
  GridCoordinate3D sizeEz;

  const GridCoordinateFP3D minHxCoordFP;
  const GridCoordinate3D minHxCoord;
  GridCoordinate3D sizeHx;

  const GridCoordinateFP3D minHyCoordFP;
  const GridCoordinate3D minHyCoord;
  GridCoordinate3D sizeHy;

  const GridCoordinateFP3D minHzCoordFP;
  const GridCoordinate3D minHzCoord;
  GridCoordinate3D sizeHz;

  GridCoordinate3D size;

public:

  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, LayoutDirection) const override;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, LayoutDirection) const override;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, LayoutDirection) const override;

  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, LayoutDirection) const override;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, LayoutDirection) const override;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, LayoutDirection) const override;

  virtual GridCoordinate3D getExSize () const override;
  virtual GridCoordinate3D getEySize () const override;
  virtual GridCoordinate3D getEzSize () const override;

  virtual GridCoordinate3D getHxSize () const override;
  virtual GridCoordinate3D getHySize () const override;
  virtual GridCoordinate3D getHzSize () const override;

  virtual GridCoordinate3D getExStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getExEnd (GridCoordinate3D) const override;
  virtual GridCoordinate3D getEyStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getEyEnd (GridCoordinate3D) const override;
  virtual GridCoordinate3D getEzStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getEzEnd (GridCoordinate3D) const override;

  virtual GridCoordinate3D getHxStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getHxEnd (GridCoordinate3D) const override;
  virtual GridCoordinate3D getHyStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getHyEnd (GridCoordinate3D) const override;
  virtual GridCoordinate3D getHzStart (GridCoordinate3D) const override;
  virtual GridCoordinate3D getHzEnd (GridCoordinate3D) const override;

  YeeGridLayout (GridCoordinate3D coordSize) :
    zeroCoordFP (0.0, 0.0, 0.0), zeroCoord (0, 0, 0),
    minExCoordFP (1.0, 0.5, 0.5), minExCoord (0, 0, 0),
    minEyCoordFP (0.5, 1.0, 0.5), minEyCoord (0, 0, 0),
    minEzCoordFP (0.5, 0.5, 1.0), minEzCoord (0, 0, 0),
    minHxCoordFP (0.5, 1.0, 1.0), minHxCoord (0, 0, 0),
    minHyCoordFP (1.0, 0.5, 1.0), minHyCoord (0, 0, 0),
    minHzCoordFP (1.0, 1.0, 0.5), minHzCoord (0, 0, 0),
    size (coordSize)
  {
    /* Ex is:
     *       1 <= x < 1 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */
    sizeEx = size - GridCoordinate3D (0, 0, 0);

    /* Ey is:
     *       0.5 <= x < 0.5 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */
    sizeEy = size - GridCoordinate3D (0, 0, 0);

    /* Ez is:
     *       0.5 <= x < 0.5 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       1 <= z < 1 + size.getZ() */
    sizeEz = size - GridCoordinate3D (0, 0, 0);

    /* Hx is:
     *       0.5 <= x < 0.5 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       1 <= z < 1 + size.getZ() */
    sizeHx = size - GridCoordinate3D (0, 0, 0);

    /* Hy is:
     *       1 <= x < 1 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       1 <= z < 1 + size.getZ() */
    sizeHy = size - GridCoordinate3D (0, 0, 0);

    /* Hz is:
     *       1 <= z < 1 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */
    sizeHz = size - GridCoordinate3D (0, 0, 0);
  }

  virtual ~YeeGridLayout ()
  {
  }
};

#endif /* GRID_LAYOUT_H */

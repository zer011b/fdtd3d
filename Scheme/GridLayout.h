#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"

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
  GridLayout () {}
  virtual ~GridLayout () {}

public:

  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;

  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, GridCoordinate3D) = 0;
};

class YeeGridLayout: public GridLayout
{
  const GridCoordinateFP3D zeroCoordFP;
  const GridCoordinate3D zeroCoord;

  const GridCoordinateFP3D minExCoordFP;
  const GridCoordinate3D minExCoord;

  const GridCoordinateFP3D minEyCoordFP;
  const GridCoordinate3D minEyCoord;

  const GridCoordinateFP3D minEzCoordFP;
  const GridCoordinate3D minEzCoord;

  const GridCoordinateFP3D minHxCoordFP;
  const GridCoordinate3D minHxCoord;

  const GridCoordinateFP3D minHyCoordFP;
  const GridCoordinate3D minHyCoord;

  const GridCoordinateFP3D minHzCoordFP;
  const GridCoordinate3D minHzCoord;

public:

  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }
  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) override
  {
    return GridCoordinate3D (0, 0, 0);
  }

  YeeGridLayout () :
    zeroCoordFP (0.0, 0.0, 0.0), zeroCoord (0, 0, 0),
    minExCoordFP (1.0, 0.5, 1.0), minExCoord (0, 0, 0),
    minEyCoordFP (0.5, 1.0, 0.5), minEyCoord (0, 0, 0),
    minEzCoordFP (0.5, 0.5, 1.0), minEzCoord (0, 0, 0),
    minHxCoordFP (0.5, 1.0, 1.0), minHxCoord (0, 0, 0),
    minHyCoordFP (1.0, 0.5, 1.0), minHyCoord (0, 0, 0),
    minHzCoordFP (1.0, 1.0, 0.5), minHzCoord (0, 0, 0),
  {
  }

  ~YeeGridLayout ()
  {
  }
};

#endif /* GRID_LAYOUT_H */

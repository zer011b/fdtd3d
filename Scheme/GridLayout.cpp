#include "GridLayout.h"

YeeGridLayout yeeLayout;

GridCoordinate3D
YeeGridLayout::getCircuitElement (GridCoordinate3D minCoord,
                                  GridCoordinateFP3D minCoordFP,
                                  GridCoordinate3D coord,
                                  LayoutDirection dir)
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minCoord) + minCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      realCoord = realCoord - GridCoordinateFP3D (0.5, 0, 0);
      break;
    }
    case LayoutDirection::RIGHT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0.5, 0, 0);
      break;
    }
    case LayoutDirection::DOWN:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0.5, 0);
      break;
    }
    case LayoutDirection::UP:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0.5, 0);
      break;
    }
    case LayoutDirection::BACK:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0, 0.5);
      break;
    }
    case LayoutDirection::FRONT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0, 0.5);
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getExCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minExCoord, minExCoordFP, coord, dir);
}

GridCoordinate3D
YeeGridLayout::getEyCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minEyCoord, minEyCoordFP, coord, dir);
}

GridCoordinate3D
YeeGridLayout::getEzCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minEzCoord, minEzCoordFP, coord, dir);
}

GridCoordinate3D
YeeGridLayout::getHxCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minHxCoord, minHxCoordFP, coord, dir);
}

GridCoordinate3D
YeeGridLayout::getHyCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minHyCoord, minHyCoordFP, coord, dir);
}

GridCoordinate3D
YeeGridLayout::getHzCircuitElement (GridCoordinate3D coord, LayoutDirection dir)
{
  return getCircuitElement (minHzCoord, minHzCoordFP, coord, dir);
}

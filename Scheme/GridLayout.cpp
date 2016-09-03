#include "GridLayout.h"

GridCoordinate3D
YeeGridLayout::getExCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minExCoord) + minExCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minHzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minHyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minHyCoordFP;
      break;
    }
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getEyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minEyCoord) + minEyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      realCoord = realCoord - GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minHzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minHxCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minHxCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getEzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minEzCoord) + minEzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      realCoord = realCoord - GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minHyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minHyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minHxCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minHxCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getHxCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minHxCoord) + minHxCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minEzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minEyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minEyCoordFP;
      break;
    }
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getHyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minHyCoord) + minHyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      realCoord = realCoord - GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minEzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minExCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0, 0.5);
      realCoord = realCoord - minExCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getHzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D realCoord = convertCoord (coord - minHzCoord) + minHzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      realCoord = realCoord - GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minEyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      realCoord = realCoord + GridCoordinateFP3D (0.5, 0, 0);
      realCoord = realCoord - minEyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      realCoord = realCoord - GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minExCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      realCoord = realCoord + GridCoordinateFP3D (0, 0.5, 0);
      realCoord = realCoord - minExCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (realCoord);
}

GridCoordinate3D
YeeGridLayout::getExSize () const
{
  return sizeEx;
}

GridCoordinate3D
YeeGridLayout::getEySize () const
{
  return sizeEy;
}

GridCoordinate3D
YeeGridLayout::getEzSize () const
{
  return sizeEz;
}

GridCoordinate3D
YeeGridLayout::getHxSize () const
{
  return sizeHx;
}

GridCoordinate3D
YeeGridLayout::getHySize () const
{
  return sizeHy;
}

GridCoordinate3D
YeeGridLayout::getHzSize () const
{
  return sizeHz;
}

GridCoordinate3D
YeeGridLayout::getExStart (GridCoordinate3D start) const
{
  return minExCoord + start + GridCoordinate3D (0, 1, 1);
}
GridCoordinate3D
YeeGridLayout::getExEnd (GridCoordinate3D end) const
{
  return minExCoord + end - GridCoordinate3D (0, 1, 1);
}

GridCoordinate3D
YeeGridLayout::getEyStart (GridCoordinate3D start) const
{
  return minEyCoord + start + GridCoordinate3D (1, 0, 1);
}
GridCoordinate3D
YeeGridLayout::getEyEnd (GridCoordinate3D end) const
{
  return minEyCoord + end - GridCoordinate3D (1, 0, 1);
}

GridCoordinate3D
YeeGridLayout::getEzStart (GridCoordinate3D start) const
{
  return minEzCoord + start + GridCoordinate3D (1, 1, 0);
}
GridCoordinate3D
YeeGridLayout::getEzEnd (GridCoordinate3D end) const
{
  return minEzCoord + end - GridCoordinate3D (1, 1, 0);
}

GridCoordinate3D
YeeGridLayout::getHxStart (GridCoordinate3D start) const
{
  return minHxCoord + start + GridCoordinate3D (0, 0, 0);
}
GridCoordinate3D
YeeGridLayout::getHxEnd (GridCoordinate3D end) const
{
  return minHxCoord + end - GridCoordinate3D (0, 1, 1);
}

GridCoordinate3D
YeeGridLayout::getHyStart (GridCoordinate3D start) const
{
  return minHyCoord + start + GridCoordinate3D (0, 0, 0);
}
GridCoordinate3D
YeeGridLayout::getHyEnd (GridCoordinate3D end) const
{
  return minHyCoord + end - GridCoordinate3D (1, 0, 1);
}

GridCoordinate3D
YeeGridLayout::getHzStart (GridCoordinate3D start) const
{
  return minHzCoord + start + GridCoordinate3D (0, 0, 0);
}
GridCoordinate3D
YeeGridLayout::getHzEnd (GridCoordinate3D end) const
{
  return minHzCoord + end - GridCoordinate3D (1, 1, 0);
}

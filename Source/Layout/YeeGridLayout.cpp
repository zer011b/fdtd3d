#include "YeeGridLayout.h"

GridCoordinate3D
YeeGridLayout::getExCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minExCoord) + minExCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

GridCoordinate3D
YeeGridLayout::getEyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minEyCoord) + minEyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

GridCoordinate3D
YeeGridLayout::getEzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minEzCoord) + minEzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

GridCoordinate3D
YeeGridLayout::getHxCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minHxCoord) + minHxCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

GridCoordinate3D
YeeGridLayout::getHyCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minHyCoord) + minHyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0, 0.5);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

GridCoordinate3D
YeeGridLayout::getHzCircuitElement (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = convertCoord (coord - minHzCoord) + minHzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + GridCoordinateFP3D (0.5, 0, 0);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + GridCoordinateFP3D (0, 0.5, 0);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    default:
    {
      UNREACHABLE;
    }
  }

  return convertCoord (coordFP);
}

bool
YeeGridLayout::isInPML (GridCoordinateFP3D realCoordFP) const
{
  GridCoordinateFP3D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
  GridCoordinateFP3D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

  /*
   * FIXME: remove floating point equality comparison
   */
  bool isInXPML = coordLeftBorderPMLFP.getX () != coordRightBorderPMLFP.getX ()
                  && (realCoordFP.getX () < coordLeftBorderPMLFP.getX ()
                      || realCoordFP.getX () >= coordRightBorderPMLFP.getX ());

  bool isInYPML = coordLeftBorderPMLFP.getY () != coordRightBorderPMLFP.getY ()
                  && (realCoordFP.getY () < coordLeftBorderPMLFP.getY ()
                      || realCoordFP.getY () >= coordRightBorderPMLFP.getY ());

  bool isInZPML = coordLeftBorderPMLFP.getZ () != coordRightBorderPMLFP.getZ ()
                  && (realCoordFP.getZ () < coordLeftBorderPMLFP.getZ ()
                      || realCoordFP.getZ () >= coordRightBorderPMLFP.getZ ());

  return isInXPML || isInYPML || isInZPML;
}

bool
YeeGridLayout::isExInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getExCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::isEyInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getEyCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::isEzInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getEzCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::isHxInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getHxCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::isHyInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getHyCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::isHzInPML (GridCoordinate3D coord) const
{
  GridCoordinateFP3D coordFP = getHzCoordFP (coord);

  return isInPML (coordFP);
}

bool
YeeGridLayout::doNeedTFSFUpdateExBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getExCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    {
      UNREACHABLE;

      break;
    }
    case LayoutDirection::DOWN:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () - 1 && coordFP.getY () < leftBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::UP:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > rightBorderFP.getY () && coordFP.getY () < rightBorderFP.getY () + 1)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::BACK:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ()
          && coordFP.getZ () > leftBorderFP.getZ () - 1 && coordFP.getZ () < leftBorderFP.getZ ())
      {
        return true;
      }

      break;
    }
    case LayoutDirection::FRONT:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ()
          && coordFP.getZ () > rightBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ () + 1)
      {
        return true;
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

bool
YeeGridLayout::doNeedTFSFUpdateEyBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getEyCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 1 && coordFP.getX () < leftBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::RIGHT:
    {
      if (coordFP.getX () > rightBorderFP.getX () && coordFP.getX () < rightBorderFP.getX () + 1
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    {
      UNREACHABLE;

      break;
    }
    case LayoutDirection::BACK:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5
          && coordFP.getZ () > leftBorderFP.getZ () - 1 && coordFP.getZ () < leftBorderFP.getZ ())
      {
        return true;
      }

      break;
    }
    case LayoutDirection::FRONT:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5
          && coordFP.getZ () > rightBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ () + 1)
      {
        return true;
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

bool
YeeGridLayout::doNeedTFSFUpdateEzBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getEzCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 1 && coordFP.getX () < leftBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::RIGHT:
    {
      if (coordFP.getX () > rightBorderFP.getX () && coordFP.getX () < rightBorderFP.getX () + 1
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::DOWN:
    {
      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 1 && coordFP.getY () < leftBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::UP:
    {
      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > rightBorderFP.getY () && coordFP.getY () < rightBorderFP.getY () + 1)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    {
      UNREACHABLE;

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

bool
YeeGridLayout::doNeedTFSFUpdateHxBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getHxCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    case LayoutDirection::RIGHT:
    {
      UNREACHABLE;

      break;
    }
    case LayoutDirection::DOWN:
    {
      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < leftBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::UP:
    {
      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > rightBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::BACK:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5
          && coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < leftBorderFP.getZ () + 0.5)
      {
        return true;
      }

      break;
    }
    case LayoutDirection::FRONT:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () && coordFP.getX () < rightBorderFP.getX ()
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5
          && coordFP.getZ () > rightBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)
      {
        return true;
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

bool
YeeGridLayout::doNeedTFSFUpdateHyBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getHyCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < leftBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::RIGHT:
    {
      if (coordFP.getX () > rightBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ())
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::DOWN:
    case LayoutDirection::UP:
    {
      UNREACHABLE;

      break;
    }
    case LayoutDirection::BACK:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ()
          && coordFP.getZ () > leftBorderFP.getZ () - 0.5 && coordFP.getZ () < leftBorderFP.getZ () + 0.5)
      {
        return true;
      }

      break;
    }
    case LayoutDirection::FRONT:
    {
      ASSERT (is3Dim);

      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () && coordFP.getY () < rightBorderFP.getY ()
          && coordFP.getZ () > rightBorderFP.getZ () - 0.5 && coordFP.getZ () < rightBorderFP.getZ () + 0.5)
      {
        return true;
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

bool
YeeGridLayout::doNeedTFSFUpdateHzBorder (GridCoordinate3D coord, LayoutDirection dir, bool is3Dim) const
{
  GridCoordinateFP3D coordFP = getHzCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < leftBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::RIGHT:
    {
      if (coordFP.getX () > rightBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::DOWN:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > leftBorderFP.getY () - 0.5 && coordFP.getY () < leftBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::UP:
    {
      if (coordFP.getX () > leftBorderFP.getX () - 0.5 && coordFP.getX () < rightBorderFP.getX () + 0.5
          && coordFP.getY () > rightBorderFP.getY () - 0.5 && coordFP.getY () < rightBorderFP.getY () + 0.5)
      {
        if (!is3Dim
            || (is3Dim && (coordFP.getZ () > leftBorderFP.getZ () && coordFP.getZ () < rightBorderFP.getZ ())))
        {
          return true;
        }
      }

      break;
    }
    case LayoutDirection::BACK:
    case LayoutDirection::FRONT:
    {
      UNREACHABLE;

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  return false;
}

FieldValue
YeeGridLayout::getExFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
}

FieldValue
YeeGridLayout::getEyFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
}

FieldValue
YeeGridLayout::getEzFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle1));
}

FieldValue
YeeGridLayout::getHxFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
}

FieldValue
YeeGridLayout::getHyFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
}

FieldValue
YeeGridLayout::getHzFromIncidentH (FieldValue valH) const
{
  return - valH * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle1));
}

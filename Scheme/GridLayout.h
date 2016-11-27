#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"
#include "Assert.h"

#ifdef CXX11_ENABLED
enum class LayoutDirection
{
  LEFT,
  RIGHT,
  DOWN,
  UP,
  BACK,
  FRONT
};
#else
class LayoutDirection
{
public:

  enum LayoutDir
  {
    LEFT,
    RIGHT,
    DOWN,
    UP,
    BACK,
    FRONT
  };

  LayoutDirection (LayoutDir new_dir)
  : dir (new_dir)
  {
  }

  operator int ()
  {
    return dir;
  }

private:

  LayoutDir dir;
};
#endif

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

  GridCoordinate3D leftBorderTotalField;
  GridCoordinate3D rightBorderTotalField;

  GridCoordinate3D leftBorderPML;
  GridCoordinate3D rightBorderPML;

public:

#ifdef CXX11_ENABLED
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
#else
  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, LayoutDirection) const;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, LayoutDirection) const;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, LayoutDirection) const;

  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, LayoutDirection) const;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, LayoutDirection) const;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, LayoutDirection) const;

  virtual GridCoordinate3D getExSize () const;
  virtual GridCoordinate3D getEySize () const;
  virtual GridCoordinate3D getEzSize () const;

  virtual GridCoordinate3D getHxSize () const;
  virtual GridCoordinate3D getHySize () const;
  virtual GridCoordinate3D getHzSize () const;

  virtual GridCoordinate3D getExStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getExEnd (GridCoordinate3D) const;
  virtual GridCoordinate3D getEyStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getEyEnd (GridCoordinate3D) const;
  virtual GridCoordinate3D getEzStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getEzEnd (GridCoordinate3D) const;

  virtual GridCoordinate3D getHxStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getHxEnd (GridCoordinate3D) const;
  virtual GridCoordinate3D getHyStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getHyEnd (GridCoordinate3D) const;
  virtual GridCoordinate3D getHzStart (GridCoordinate3D) const;
  virtual GridCoordinate3D getHzEnd (GridCoordinate3D) const;
#endif

  GridCoordinateFP3D getZeroCoordFP () const
  {
    return zeroCoordFP;
  }

  GridCoordinateFP3D getMinEzCoordFP () const
  {
    return minEzCoordFP;
  }

  GridCoordinateFP3D getMinHyCoordFP () const
  {
    return minHyCoordFP;
  }

  GridCoordinateFP3D getEzRealCoord (GridCoordinate3D coord) const
  {
    return convertCoord (coord - minEzCoord) + minEzCoordFP;
  }

  GridCoordinateFP3D getHxRealCoord (GridCoordinate3D coord) const
  {
    return convertCoord (coord - minHxCoord) + minHxCoordFP;
  }

  GridCoordinateFP3D getHyRealCoord (GridCoordinate3D coord) const
  {
    return convertCoord (coord - minHyCoord) + minHyCoordFP;
  }

  bool isInPML (GridCoordinateFP3D realCoordFP) const
  {
    GridCoordinateFP3D realCoordLeftBorderPML = convertCoord (leftBorderPML) + zeroCoordFP;
    GridCoordinateFP3D realCoordRightBorderPML = convertCoord (rightBorderPML) + zeroCoordFP;

    bool isInXPML = realCoordLeftBorderPML.getX () != realCoordRightBorderPML.getX ()
                    && (realCoordFP.getX () < realCoordLeftBorderPML.getX ()
                        || realCoordFP.getX () >= realCoordRightBorderPML.getX ());

    bool isInYPML = realCoordLeftBorderPML.getY () != realCoordRightBorderPML.getY ()
                    && (realCoordFP.getY () < realCoordLeftBorderPML.getY ()
                        || realCoordFP.getY () >= realCoordRightBorderPML.getY ());

    bool isInZPML = realCoordLeftBorderPML.getZ () != realCoordRightBorderPML.getZ ()
                    && (realCoordFP.getZ () < realCoordLeftBorderPML.getZ ()
                        || realCoordFP.getZ () >= realCoordRightBorderPML.getZ ());

    return isInXPML || isInYPML || isInZPML;
  }

  bool isEzInPML (GridCoordinate3D coord) const
  {
    GridCoordinateFP3D realCoordFP = getEzRealCoord (coord);

    return isInPML (realCoordFP);
  }

  bool isHxInPML (GridCoordinate3D coord) const
  {
    GridCoordinateFP3D realCoordFP = getHxRealCoord (coord);

    return isInPML (realCoordFP);
  }

  bool isHyInPML (GridCoordinate3D coord) const
  {
    GridCoordinateFP3D realCoordFP = getHyRealCoord (coord);

    return isInPML (realCoordFP);
  }

  GridCoordinate3D getLeftBorderPML () const
  {
    return leftBorderPML;
  }

  GridCoordinate3D getRightBorderPML () const
  {
    return rightBorderPML;
  }

  bool doNeedTFSFUpdateEzBorder (GridCoordinate3D coord, LayoutDirection dir) const
  {
    /*
     * FIXME: 3d not implemented
     */
    GridCoordinateFP3D realCoord = getEzRealCoord (coord);

    GridCoordinateFP3D leftBorder = zeroCoordFP + convertCoord (leftBorderTotalField);
    GridCoordinateFP3D rightBorder = zeroCoordFP + convertCoord (rightBorderTotalField);

    switch (dir)
    {
      case LayoutDirection::LEFT:
      {
        if (realCoord.getX () > leftBorder.getX () - 1 && realCoord.getX () < leftBorder.getX ()
            && realCoord.getY () > leftBorder.getY () && realCoord.getY () < rightBorder.getY ())
        {
          return true;
        }

        break;
      }
      case LayoutDirection::RIGHT:
      {
        if (realCoord.getX () > rightBorder.getX () && realCoord.getX () < rightBorder.getX () + 1
            && realCoord.getY () > leftBorder.getY () && realCoord.getY () < rightBorder.getY ())
        {
          return true;
        }

        break;
      }
      case LayoutDirection::DOWN:
      {
        if (realCoord.getX () > leftBorder.getX () && realCoord.getX () < rightBorder.getX ()
            && realCoord.getY () > leftBorder.getY () - 1 && realCoord.getY () < leftBorder.getY ())
        {
          return true;
        }

        break;
      }
      case LayoutDirection::UP:
      {
        if (realCoord.getX () > leftBorder.getX () && realCoord.getX () < rightBorder.getX ()
            && realCoord.getY () > rightBorder.getY () && realCoord.getY () < rightBorder.getY () + 1)
        {
          return true;
        }

        break;
      }
      case LayoutDirection::BACK:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      case LayoutDirection::FRONT:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return false;
  }

  bool doNeedTFSFUpdateHxBorder (GridCoordinate3D coord, LayoutDirection dir) const
  {
    /*
     * FIXME: 3d not implemented
     */
    GridCoordinateFP3D realCoord = getHxRealCoord (coord);

    GridCoordinateFP3D leftBorder = zeroCoordFP + convertCoord (leftBorderTotalField);
    GridCoordinateFP3D rightBorder = zeroCoordFP + convertCoord (rightBorderTotalField);

    switch (dir)
    {
      case LayoutDirection::LEFT:
      {
        break;
      }
      case LayoutDirection::RIGHT:
      {
        break;
      }
      case LayoutDirection::DOWN:
      {
        if (realCoord.getX () > leftBorder.getX () && realCoord.getX () < rightBorder.getX ()
            && realCoord.getY () > leftBorder.getY () - 0.5 && realCoord.getY () < leftBorder.getY () + 0.5)
        {
          return true;
        }

        break;
      }
      case LayoutDirection::UP:
      {
        if (realCoord.getX () > leftBorder.getX () && realCoord.getX () < rightBorder.getX ()
            && realCoord.getY () > rightBorder.getY () - 0.5 && realCoord.getY () < rightBorder.getY () + 0.5)
        {
          return true;
        }

        break;
      }
      case LayoutDirection::BACK:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      case LayoutDirection::FRONT:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return false;
  }

  bool doNeedTFSFUpdateHyBorder (GridCoordinate3D coord, LayoutDirection dir) const
  {
    /*
     * FIXME: 3d not implemented
     */
    GridCoordinateFP3D realCoord = getHyRealCoord (coord);

    GridCoordinateFP3D leftBorder = zeroCoordFP + convertCoord (leftBorderTotalField);
    GridCoordinateFP3D rightBorder = zeroCoordFP + convertCoord (rightBorderTotalField);

    switch (dir)
    {
      case LayoutDirection::LEFT:
      {
        if (realCoord.getX () > leftBorder.getX () - 0.5 && realCoord.getX () < leftBorder.getX () + 0.5
            && realCoord.getY () > leftBorder.getY () && realCoord.getY () < rightBorder.getY ())
        {
          return true;
        }

        break;
      }
      case LayoutDirection::RIGHT:
      {
        if (realCoord.getX () > rightBorder.getX () - 0.5 && realCoord.getX () < rightBorder.getX () + 0.5
            && realCoord.getY () > leftBorder.getY () && realCoord.getY () < rightBorder.getY ())
        {
          return true;
        }

        break;
      }
      case LayoutDirection::DOWN:
      {
        break;
      }
      case LayoutDirection::UP:
      {
        break;
      }
      case LayoutDirection::BACK:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      case LayoutDirection::FRONT:
      {
        ASSERT_MESSAGE ("Unimplemented");
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return false;
  }

  YeeGridLayout (GridCoordinate3D coordSize, GridCoordinate3D sizePML, GridCoordinate3D sizeScatteredZone) :
    zeroCoordFP (0.0, 0.0, 0.0), zeroCoord (0, 0, 0),
    minExCoordFP (1.0, 0.5, 0.5), minExCoord (0, 0, 0),
    minEyCoordFP (0.5, 1.0, 0.5), minEyCoord (0, 0, 0),
    minEzCoordFP (0.5, 0.5, 1.0), minEzCoord (0, 0, 0),
    minHxCoordFP (0.5, 1.0, 1.0), minHxCoord (0, 0, 0),
    minHyCoordFP (1.0, 0.5, 1.0), minHyCoord (0, 0, 0),
    minHzCoordFP (1.0, 1.0, 0.5), minHzCoord (0, 0, 0),
    size (coordSize),
    leftBorderTotalField (sizeScatteredZone),
    rightBorderTotalField (coordSize - sizeScatteredZone),
    leftBorderPML (sizePML),
    rightBorderPML (coordSize - sizePML)
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

#ifndef YEE_GRID_LAYOUT_H
#define YEE_GRID_LAYOUT_H

#include "Approximation.h"
#include "Assert.h"
#include "Grid.h"
#include "GridLayout.h"
#include "PhysicsConst.h"

#include <cmath>

/**
 * Yee grid layout which specifies how field components are placed in space
 *
 * FIXME: add link to docs with description of yee layout
 *
 * FIXME: consider removing minCoord which is always currently 0
 *
 * FIXME: make template by type of coordinate
 *
 * FIXME: inline getter/setter
 */
class YeeGridLayout: public GridLayout
{
protected:

  const GridCoordinate3D zeroCoord; /**< Zero coordinate of grid */
  const GridCoordinate3D minEpsCoord; /**< Minimum epsilon coordinate */
  const GridCoordinate3D minMuCoord; /**< Minimum mu coordinate */
  const GridCoordinate3D minExCoord; /**< Minimum Ex field coordinate */
  const GridCoordinate3D minEyCoord; /**< Minimum Ey field coordinate */
  const GridCoordinate3D minEzCoord; /**< Minimum Ez field coordinate */
  const GridCoordinate3D minHxCoord; /**< Minimum Hx field coordinate */
  const GridCoordinate3D minHyCoord; /**< Minimum Hy field coordinate */
  const GridCoordinate3D minHzCoord; /**< Minimum Hz field coordinate */

  const GridCoordinateFP3D zeroCoordFP; /**< Real zero coordinate of grid (corresponding to zeroCoord) */
  const GridCoordinateFP3D minEpsCoordFP; /**< Minimum real epsilon coordinate (corresponding to minEpsCoord) */
  const GridCoordinateFP3D minMuCoordFP; /**< Minimum real mu coordinate (corresponding to minMuCoord) */
  const GridCoordinateFP3D minExCoordFP; /**< Minimum real Ex field coordinate (corresponding to minExCoord) */
  const GridCoordinateFP3D minEyCoordFP; /**< Minimum real Ey field coordinate (corresponding to minEyCoord) */
  const GridCoordinateFP3D minEzCoordFP; /**< Minimum real Ez field coordinate (corresponding to minEzCoord) */
  const GridCoordinateFP3D minHxCoordFP; /**< Minimum real Hx field coordinate (corresponding to minHxCoord) */
  const GridCoordinateFP3D minHyCoordFP; /**< Minimum real Hy field coordinate (corresponding to minHyCoord) */
  const GridCoordinateFP3D minHzCoordFP; /**< Minimum real Hz field coordinate (corresponding to minHzCoord) */

  const GridCoordinate3D size; /**< Size of grids */
  const GridCoordinate3D sizeEps; /**< Size of epsilon grid */
  const GridCoordinate3D sizeMu; /**< Size of mu grid */
  const GridCoordinate3D sizeEx; /**< size of Ex field */
  const GridCoordinate3D sizeEy; /**< size of Ey field */
  const GridCoordinate3D sizeEz; /**< size of Ez field */
  const GridCoordinate3D sizeHx; /**< size of Hx field */
  const GridCoordinate3D sizeHy; /**< size of Hy field */
  const GridCoordinate3D sizeHz; /**< size of Hz field */

  const GridCoordinate3D leftBorderPML; /**< Coordinate of left border of PML */
  const GridCoordinate3D rightBorderPML; /**< Coordinate of right border of PML */

  const GridCoordinate3D leftBorderTotalField; /**< Coordinate of left border of TF/SF */
  const GridCoordinate3D rightBorderTotalField; /**< Coordinate of right border of TF/SF */
  const GridCoordinateFP3D zeroIncCoordFP; /**< Real coordinate corresponding to zero coordinate of auxiliary grid
                                            *   for incident wave */

  const FPValue incidentWaveAngle1; /**< Teta incident wave angle */
  const FPValue incidentWaveAngle2; /**< Phi incident wave angle */
  const FPValue incidentWaveAngle3; /**< Psi incident wave angle */

  const bool isDoubleMaterialPrecision;

public:

  /*
   * Get coordinate of circut field component
   */
  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, LayoutDirection) const CXX11_OVERRIDE_FINAL;

  /*
   * Get size of field component grid
   */
  virtual GridCoordinate3D getEpsSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeEps;
  }
  virtual GridCoordinate3D getMuSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeEps;
  }
  virtual GridCoordinate3D getExSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeEx;
  }
  virtual GridCoordinate3D getEySize () const CXX11_OVERRIDE_FINAL
  {
    return sizeEy;
  }
  virtual GridCoordinate3D getEzSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeEz;
  }
  virtual GridCoordinate3D getHxSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeHx;
  }
  virtual GridCoordinate3D getHySize () const CXX11_OVERRIDE_FINAL
  {
    return sizeHy;
  }
  virtual GridCoordinate3D getHzSize () const CXX11_OVERRIDE_FINAL
  {
    return sizeHz;
  }

  virtual GridCoordinate3D getSizePML () const CXX11_OVERRIDE_FINAL
  {
    return leftBorderPML;
  }
  virtual GridCoordinate3D getSizeTFSF () const CXX11_OVERRIDE_FINAL
  {
    return leftBorderTotalField;
  }

  /*
   * Get start coordinate of field component
   */
  virtual GridCoordinate3D getExStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minExCoord + GridCoordinate3D (1, 1, 1);
  }
  virtual GridCoordinate3D getEyStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minEyCoord + GridCoordinate3D (1, 1, 1);
  }
  virtual GridCoordinate3D getEzStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minEzCoord + GridCoordinate3D (1, 1, 1);
  }
  virtual GridCoordinate3D getHxStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minHxCoord + GridCoordinate3D (1, 1, 1);
  }
  virtual GridCoordinate3D getHyStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minHyCoord + GridCoordinate3D (1, 1, 1);
  }
  virtual GridCoordinate3D getHzStartDiff () const CXX11_OVERRIDE_FINAL
  {
    return minHzCoord + GridCoordinate3D (1, 1, 1);
  }

  /*
   * Get end coordinate of field component
   */
  virtual GridCoordinate3D getExEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minExCoord;
  }
  virtual GridCoordinate3D getEyEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minEyCoord;
  }
  virtual GridCoordinate3D getEzEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minEzCoord;
  }
  virtual GridCoordinate3D getHxEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minHxCoord;
  }
  virtual GridCoordinate3D getHyEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minHyCoord;
  }
  virtual GridCoordinate3D getHzEndDiff () const CXX11_OVERRIDE_FINAL
  {
    return GridCoordinate3D (1, 1, 1) - minHzCoord;
  }

  /*
   * Get minimum coordinate of field component
   */
  virtual GridCoordinate3D getZeroCoord () const CXX11_OVERRIDE_FINAL
  {
    return zeroCoord;
  }
  virtual GridCoordinate3D getMinEpsCoord () const CXX11_OVERRIDE_FINAL
  {
    return minEpsCoord;
  }
  virtual GridCoordinate3D getMinMuCoord () const CXX11_OVERRIDE_FINAL
  {
    return minMuCoord;
  }
  virtual GridCoordinate3D getMinExCoord () const CXX11_OVERRIDE_FINAL
  {
    return minExCoord;
  }
  virtual GridCoordinate3D getMinEyCoord () const CXX11_OVERRIDE_FINAL
  {
    return minEyCoord;
  }
  virtual GridCoordinate3D getMinEzCoord () const CXX11_OVERRIDE_FINAL
  {
    return minEzCoord;
  }
  virtual GridCoordinate3D getMinHxCoord () const CXX11_OVERRIDE_FINAL
  {
    return minHxCoord;
  }
  virtual GridCoordinate3D getMinHyCoord () const CXX11_OVERRIDE_FINAL
  {
    return minHyCoord;
  }
  virtual GridCoordinate3D getMinHzCoord () const CXX11_OVERRIDE_FINAL
  {
    return minHzCoord;
  }

  /*
   * Get minimum real coordinate of field component
   */
  virtual GridCoordinateFP3D getZeroCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return zeroCoordFP;
  }
  virtual GridCoordinateFP3D getMinEpsCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minEpsCoordFP;
  }
  virtual GridCoordinateFP3D getMinMuCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minMuCoordFP;
  }
  virtual GridCoordinateFP3D getMinExCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minExCoordFP;
  }
  virtual GridCoordinateFP3D getMinEyCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minEyCoordFP;
  }
  virtual GridCoordinateFP3D getMinEzCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minEzCoordFP;
  }
  virtual GridCoordinateFP3D getMinHxCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minHxCoordFP;
  }
  virtual GridCoordinateFP3D getMinHyCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minHyCoordFP;
  }
  virtual GridCoordinateFP3D getMinHzCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return minHzCoordFP;
  }

  /*
   * Get real coordinate of field component by its coordinate
   */
  virtual GridCoordinateFP3D getEpsCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEpsCoord) + minEpsCoordFP;
  }
  virtual GridCoordinateFP3D getMuCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minMuCoord) + minMuCoordFP;
  }
  virtual GridCoordinateFP3D getExCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minExCoord) + minExCoordFP;
  }
  virtual GridCoordinateFP3D getEyCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEyCoord) + minEyCoordFP;
  }
  virtual GridCoordinateFP3D getEzCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEzCoord) + minEzCoordFP;
  }
  virtual GridCoordinateFP3D getHxCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHxCoord) + minHxCoordFP;
  }
  virtual GridCoordinateFP3D getHyCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHyCoord) + minHyCoordFP;
  }
  virtual GridCoordinateFP3D getHzCoordFP (GridCoordinate3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHzCoord) + minHzCoordFP;
  }

  /*
   * Get coordinate of field component by its real coordinate
   */
  virtual GridCoordinate3D getEpsCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEpsCoordFP) + minEpsCoord;
  }
  virtual GridCoordinate3D getMuCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minMuCoordFP) + minMuCoord;
  }
  virtual GridCoordinate3D getExCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minExCoordFP) + minExCoord;
  }
  virtual GridCoordinate3D getEyCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEyCoordFP) + minEyCoord;
  }
  virtual GridCoordinate3D getEzCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minEzCoordFP) + minEzCoord;
  }
  virtual GridCoordinate3D getHxCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHxCoordFP) + minHxCoord;
  }
  virtual GridCoordinate3D getHyCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHyCoordFP) + minHyCoord;
  }
  virtual GridCoordinate3D getHzCoord (GridCoordinateFP3D coord) const CXX11_OVERRIDE_FINAL
  {
    return convertCoord (coord - minHzCoordFP) + minHzCoord;
  }

  /*
   * PML
   */
  virtual GridCoordinate3D getLeftBorderPML () const CXX11_OVERRIDE_FINAL
  {
    return leftBorderPML;
  }
  virtual GridCoordinate3D getRightBorderPML () const CXX11_OVERRIDE_FINAL
  {
    return rightBorderPML;
  }
  virtual bool isInPML (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isExInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isEyInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isEzInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isHxInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isHyInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual bool isHzInPML (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;

  /*
   * Total field / scattered field
   */
  virtual GridCoordinate3D getLeftBorderTFSF () const CXX11_OVERRIDE_FINAL
  {
    return leftBorderTotalField;
  }
  virtual GridCoordinate3D getRightBorderTFSF () const CXX11_OVERRIDE_FINAL
  {
    return rightBorderTotalField;
  }
  virtual GridCoordinateFP3D getZeroIncCoordFP () const CXX11_OVERRIDE_FINAL
  {
    return zeroIncCoordFP;
  }
  virtual bool doNeedTFSFUpdateExBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateEyBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateEzBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHxBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHyBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHzBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;

  virtual FPValue getIncidentWaveAngle1 () const CXX11_OVERRIDE_FINAL
  {
    return incidentWaveAngle1;
  }
  virtual FPValue getIncidentWaveAngle2 () const CXX11_OVERRIDE_FINAL
  {
    return incidentWaveAngle2;
  }
  virtual FPValue getIncidentWaveAngle3 () const CXX11_OVERRIDE_FINAL
  {
    return incidentWaveAngle3;
  }

  virtual FieldValue getExFromIncidentE (FieldValue) const CXX11_OVERRIDE_FINAL;
  virtual FieldValue getEyFromIncidentE (FieldValue) const CXX11_OVERRIDE_FINAL;
  virtual FieldValue getEzFromIncidentE (FieldValue) const CXX11_OVERRIDE_FINAL;
  virtual FieldValue getHxFromIncidentH (FieldValue) const CXX11_OVERRIDE_FINAL;
  virtual FieldValue getHyFromIncidentH (FieldValue) const CXX11_OVERRIDE_FINAL;
  virtual FieldValue getHzFromIncidentH (FieldValue) const CXX11_OVERRIDE_FINAL;

  /**
   * Constructor of Yee grid
   */
  YeeGridLayout (GridCoordinate3D coordSize,
                 GridCoordinate3D sizePML,
                 GridCoordinate3D sizeScatteredZone,
                 FPValue incWaveAngle1, /**< teta */
                 FPValue incWaveAngle2, /**< phi */
                 FPValue incWaveAngle3, /**< psi */
                 bool doubleMaterialPrecision)
    : zeroCoord (0, 0, 0)
  , minEpsCoord (0, 0, 0)
  , minMuCoord (0, 0, 0)
  , minExCoord (0, 0, 0)
  , minEyCoord (0, 0, 0)
  , minEzCoord (0, 0, 0)
  , minHxCoord (0, 0, 0)
  , minHyCoord (0, 0, 0)
  , minHzCoord (0, 0, 0)
  , minEpsCoordFP (0.5, 0.5, 0.5)
  , minMuCoordFP (0.5, 0.5, 0.5)
  , zeroCoordFP (0.0, 0.0, 0.0)
  , minExCoordFP (1.0, 0.5, 0.5)
  , minEyCoordFP (0.5, 1.0, 0.5)
  , minEzCoordFP (0.5, 0.5, 1.0)
  , minHxCoordFP (0.5, 1.0, 1.0)
  , minHyCoordFP (1.0, 0.5, 1.0)
  , minHzCoordFP (1.0, 1.0, 0.5)
  , size (coordSize)
  , sizeEps (coordSize * (doubleMaterialPrecision ? 2 : 1))
  , sizeMu (coordSize * (doubleMaterialPrecision ? 2 : 1))
  , sizeEx (coordSize)
  , sizeEy (coordSize)
  , sizeEz (coordSize)
  , sizeHx (coordSize)
  , sizeHy (coordSize)
  , sizeHz (coordSize)
  , leftBorderPML (sizePML)
  , rightBorderPML (coordSize - sizePML)
  , leftBorderTotalField (sizeScatteredZone)
  , rightBorderTotalField (coordSize - sizeScatteredZone)
  , incidentWaveAngle1 (incWaveAngle1)
  , incidentWaveAngle2 (incWaveAngle2)
  , incidentWaveAngle3 (incWaveAngle3)
  , isDoubleMaterialPrecision (doubleMaterialPrecision)
  {
    ASSERT (coordSize.getX () > 0);

    FPValue incCoordX = leftBorderTotalField.getX () - 2.5 * sin (incWaveAngle1) * cos (incWaveAngle2);
    FPValue incCoordY = 0.0;
    FPValue incCoordZ = 0.0;

    if (coordSize.getY () == 0)
    {
      ASSERT (coordSize.getZ () == 0);
    }
    else
    {
      incCoordY = leftBorderTotalField.getY () - 2.5 * sin (incWaveAngle1) * sin (incWaveAngle2);

      if (coordSize.getZ () != 0)
      {
        incCoordZ = leftBorderTotalField.getZ () - 2.5 * cos (incWaveAngle1);
      }
    }

    ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
    ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
    /* Ex is:
     *       1 <= x < 1 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */

    /* Ey is:
     *       0.5 <= x < 0.5 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */

    /* Ez is:
     *       0.5 <= x < 0.5 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       1 <= z < 1 + size.getZ() */

    /* Hx is:
     *       0.5 <= x < 0.5 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       1 <= z < 1 + size.getZ() */

    /* Hy is:
     *       1 <= x < 1 + size.getx()
     *       0.5 <= y < 0.5 + size.getY()
     *       1 <= z < 1 + size.getZ() */

    /* Hz is:
     *       1 <= z < 1 + size.getx()
     *       1 <= y < 1 + size.getY()
     *       0.5 <= z < 0.5 + size.getZ() */
  } /* YeeGridLayout */

  virtual ~YeeGridLayout ()
  {
  } /* ~YeeGridLayout */

  template <class TCoord>
  FPValue getApproximateMaterial (Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D);
  template <class TCoord>
  FPValue getApproximateMaterial (Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  template <class TCoord>
  FPValue getApproximateMaterial (Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D,
                                  GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  template <class TCoord>
  FPValue getApproximateMetaMaterial (Grid<TCoord> *, Grid<TCoord> *, Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D,
                                      FPValue &, FPValue &);
  template <class TCoord>
  FPValue getApproximateMetaMaterial (Grid<TCoord> *, Grid<TCoord> *, Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D,
                                      GridCoordinate3D, GridCoordinate3D, FPValue &, FPValue &);
  template <class TCoord>
  FPValue getApproximateMetaMaterial (Grid<TCoord> *, Grid<TCoord> *, Grid<TCoord> *, GridCoordinate3D, GridCoordinate3D,
                                      GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D,
                                      GridCoordinate3D, GridCoordinate3D, FPValue &, FPValue &);

  template <class TCoord>
  FPValue getMetaMaterial (GridCoordinate3D &, GridType, Grid<TCoord> *, GridType, Grid<TCoord> *, GridType, Grid<TCoord> *,
                           GridType, FPValue &, FPValue &);
  template <class TCoord>
  FPValue getMaterial (GridCoordinate3D &, GridType, Grid<TCoord> *, GridType);

  bool getIsDoubleMaterialPrecision () const
  {
    return isDoubleMaterialPrecision;
  }
}; /* YeeGridLayout */

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMaterial (Grid<TCoord> *gridMaterial,
                                       GridCoordinate3D coord1,
                                       GridCoordinate3D coord2)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);

  return Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                             Approximation::getMaterial (val2));
}

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMaterial (Grid<TCoord> *gridMaterial,
                                       GridCoordinate3D coord1,
                                       GridCoordinate3D coord2,
                                       GridCoordinate3D coord3,
                                       GridCoordinate3D coord4)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue* val3 = gridMaterial->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue* val4 = gridMaterial->getFieldPointValueByAbsolutePos (coord4);

  return Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                             Approximation::getMaterial (val2),
                                             Approximation::getMaterial (val3),
                                             Approximation::getMaterial (val4));
}

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMaterial (Grid<TCoord> *gridMaterial,
                                       GridCoordinate3D coord1,
                                       GridCoordinate3D coord2,
                                       GridCoordinate3D coord3,
                                       GridCoordinate3D coord4,
                                       GridCoordinate3D coord5,
                                       GridCoordinate3D coord6,
                                       GridCoordinate3D coord7,
                                       GridCoordinate3D coord8)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue* val3 = gridMaterial->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue* val4 = gridMaterial->getFieldPointValueByAbsolutePos (coord4);
  FieldPointValue* val5 = gridMaterial->getFieldPointValueByAbsolutePos (coord5);
  FieldPointValue* val6 = gridMaterial->getFieldPointValueByAbsolutePos (coord6);
  FieldPointValue* val7 = gridMaterial->getFieldPointValueByAbsolutePos (coord7);
  FieldPointValue* val8 = gridMaterial->getFieldPointValueByAbsolutePos (coord8);

  return Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                             Approximation::getMaterial (val2),
                                             Approximation::getMaterial (val3),
                                             Approximation::getMaterial (val4),
                                             Approximation::getMaterial (val5),
                                             Approximation::getMaterial (val6),
                                             Approximation::getMaterial (val7),
                                             Approximation::getMaterial (val8));
}

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMetaMaterial (Grid<TCoord> *gridMaterial,
                                           Grid<TCoord> *gridMaterialOmega,
                                           Grid<TCoord> *gridMaterialGamma,
                                           GridCoordinate3D coord1,
                                           GridCoordinate3D coord2,
                                           FPValue &omega,
                                           FPValue &gamma)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);

  FPValue material = Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                                         Approximation::getMaterial (val2));

  FieldPointValue *val3 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val4 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord2);

  FieldPointValue *val5 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val6 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord2);

  Approximation::approximateDrudeModel (omega,
                                        gamma,
                                        Approximation::getMaterial (val1),
                                        Approximation::getMaterial (val2),
                                        Approximation::getMaterial (val3),
                                        Approximation::getMaterial (val4),
                                        Approximation::getMaterial (val5),
                                        Approximation::getMaterial (val6));

  return material;
}

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMetaMaterial (Grid<TCoord> *gridMaterial,
                                           Grid<TCoord> *gridMaterialOmega,
                                           Grid<TCoord> *gridMaterialGamma,
                                           GridCoordinate3D coord1,
                                           GridCoordinate3D coord2,
                                           GridCoordinate3D coord3,
                                           GridCoordinate3D coord4,
                                           FPValue &omega,
                                           FPValue &gamma)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue* val3 = gridMaterial->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue* val4 = gridMaterial->getFieldPointValueByAbsolutePos (coord4);

  FPValue material = Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                                         Approximation::getMaterial (val2),
                                                         Approximation::getMaterial (val3),
                                                         Approximation::getMaterial (val4));

  FieldPointValue *val5 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val6 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue *val7 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue *val8 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord4);

  FieldPointValue *val9 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val10 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue *val11 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue *val12 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord4);

  Approximation::approximateDrudeModel (omega,
                                        gamma,
                                        Approximation::getMaterial (val1),
                                        Approximation::getMaterial (val2),
                                        Approximation::getMaterial (val3),
                                        Approximation::getMaterial (val4),
                                        Approximation::getMaterial (val5),
                                        Approximation::getMaterial (val6),
                                        Approximation::getMaterial (val7),
                                        Approximation::getMaterial (val8),
                                        Approximation::getMaterial (val9),
                                        Approximation::getMaterial (val10),
                                        Approximation::getMaterial (val11),
                                        Approximation::getMaterial (val12));

  return material;
}

template <class TCoord>
FPValue
YeeGridLayout::getApproximateMetaMaterial (Grid<TCoord> *gridMaterial,
                                           Grid<TCoord> *gridMaterialOmega,
                                           Grid<TCoord> *gridMaterialGamma,
                                           GridCoordinate3D coord1,
                                           GridCoordinate3D coord2,
                                           GridCoordinate3D coord3,
                                           GridCoordinate3D coord4,
                                           GridCoordinate3D coord5,
                                           GridCoordinate3D coord6,
                                           GridCoordinate3D coord7,
                                           GridCoordinate3D coord8,
                                           FPValue &omega,
                                           FPValue &gamma)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue* val3 = gridMaterial->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue* val4 = gridMaterial->getFieldPointValueByAbsolutePos (coord4);
  FieldPointValue* val5 = gridMaterial->getFieldPointValueByAbsolutePos (coord5);
  FieldPointValue* val6 = gridMaterial->getFieldPointValueByAbsolutePos (coord6);
  FieldPointValue* val7 = gridMaterial->getFieldPointValueByAbsolutePos (coord7);
  FieldPointValue* val8 = gridMaterial->getFieldPointValueByAbsolutePos (coord8);

  FPValue material = Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                                         Approximation::getMaterial (val2),
                                                         Approximation::getMaterial (val3),
                                                         Approximation::getMaterial (val4),
                                                         Approximation::getMaterial (val5),
                                                         Approximation::getMaterial (val6),
                                                         Approximation::getMaterial (val7),
                                                         Approximation::getMaterial (val8));

  FieldPointValue *val9 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val10 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue *val11 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue *val12 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord4);
  FieldPointValue *val13 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord5);
  FieldPointValue *val14 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord6);
  FieldPointValue *val15 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord7);
  FieldPointValue *val16 = gridMaterialOmega->getFieldPointValueByAbsolutePos (coord8);

  FieldPointValue *val17 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue *val18 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord2);
  FieldPointValue *val19 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord3);
  FieldPointValue *val20 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord4);
  FieldPointValue *val21 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord5);
  FieldPointValue *val22 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord6);
  FieldPointValue *val23 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord7);
  FieldPointValue *val24 = gridMaterialGamma->getFieldPointValueByAbsolutePos (coord8);

  Approximation::approximateDrudeModel (omega,
                                        gamma,
                                        Approximation::getMaterial (val1),
                                        Approximation::getMaterial (val2),
                                        Approximation::getMaterial (val3),
                                        Approximation::getMaterial (val4),
                                        Approximation::getMaterial (val5),
                                        Approximation::getMaterial (val6),
                                        Approximation::getMaterial (val7),
                                        Approximation::getMaterial (val8),
                                        Approximation::getMaterial (val9),
                                        Approximation::getMaterial (val10),
                                        Approximation::getMaterial (val11),
                                        Approximation::getMaterial (val12),
                                        Approximation::getMaterial (val13),
                                        Approximation::getMaterial (val14),
                                        Approximation::getMaterial (val15),
                                        Approximation::getMaterial (val16),
                                        Approximation::getMaterial (val17),
                                        Approximation::getMaterial (val18),
                                        Approximation::getMaterial (val19),
                                        Approximation::getMaterial (val20),
                                        Approximation::getMaterial (val21),
                                        Approximation::getMaterial (val22),
                                        Approximation::getMaterial (val23),
                                        Approximation::getMaterial (val24));

  return material;
}

template <class TCoord>
FPValue
YeeGridLayout::getMetaMaterial (GridCoordinate3D &posAbs,
                                GridType typeOfField,
                                Grid<TCoord> *gridMaterial,
                                GridType typeOfMaterial,
                                Grid<TCoord> *gridMaterialOmega,
                                GridType typeOfMaterialOmega,
                                Grid<TCoord> *gridMaterialGamma,
                                GridType typeOfMaterialGamma,
                                FPValue &omega,
                                FPValue &gamma)
{
  GridCoordinate3D absPos11;
  GridCoordinate3D absPos12;
  GridCoordinate3D absPos21;
  GridCoordinate3D absPos22;

  GridCoordinate3D absPos31;
  GridCoordinate3D absPos32;
  GridCoordinate3D absPos41;
  GridCoordinate3D absPos42;

  switch (typeOfField)
  {
    case GridType::EX:
    case GridType::DX:
    {
      GridCoordinateFP3D realCoord = getExCoordFP (posAbs);

      /*
       * TODO: add separate for all material grids
       */
      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 0, 0);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 1, 0);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 1, 0);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 1, 1);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 0, 1);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 1, 1);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0));
      }

      break;
    }
    case GridType::EY:
    case GridType::DY:
    {
      GridCoordinateFP3D realCoord = getEyCoordFP (posAbs);

      /*
       * TODO: add separate for all material grids
       */
      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 1, 0);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 1, 0);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 0, 0);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 1, 1);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 1, 1);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 0, 1);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 0, 1);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0));
      }

      break;
    }
    case GridType::EZ:
    case GridType::DZ:
    {
      GridCoordinateFP3D realCoord = getEzCoordFP (posAbs);

      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 0, 1);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 1, 1);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 0, 1);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 1, 1);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 0, 0);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 1, 0);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 1, 0);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5));
      }

      break;
    }
    case GridType::HX:
    case GridType::BX:
    {
      GridCoordinateFP3D realCoord = getHxCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () - 0.5);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () + 0.5);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () - 0.5);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () + 0.5);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () - 0.5);
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () + 0.5);
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () - 0.5);
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () + 0.5);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HY:
    case GridType::BY:
    {
      GridCoordinateFP3D realCoord = getHyCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HZ:
    case GridType::BZ:
    {
      GridCoordinateFP3D realCoord = getHzCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT ((typeOfMaterialOmega == GridType::OMEGAPE && typeOfMaterialGamma == GridType::GAMMAE)
          || (typeOfMaterialOmega == GridType::OMEGAPM && typeOfMaterialGamma == GridType::GAMMAM));

  if (isDoubleMaterialPrecision)
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getApproximateMetaMaterial (gridMaterial, gridMaterialOmega, gridMaterialGamma, absPos11, absPos12, absPos21, absPos22, absPos31, absPos32, absPos41, absPos42, omega, gamma);
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getApproximateMetaMaterial (gridMaterial, gridMaterialOmega, gridMaterialGamma, absPos11, absPos12, omega, gamma);
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return getApproximateMetaMaterial (gridMaterial, gridMaterialOmega, gridMaterialGamma, absPos11, absPos12, absPos21, absPos22, omega, gamma);
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
}

template <class TCoord>
FPValue
YeeGridLayout::getMaterial (GridCoordinate3D &posAbs,
                            GridType typeOfField,
                            Grid<TCoord> *gridMaterial,
                            GridType typeOfMaterial)
{
  GridCoordinate3D absPos11;
  GridCoordinate3D absPos12;
  GridCoordinate3D absPos21;
  GridCoordinate3D absPos22;

  GridCoordinate3D absPos31;
  GridCoordinate3D absPos32;
  GridCoordinate3D absPos41;
  GridCoordinate3D absPos42;

  switch (typeOfField)
  {
    case GridType::EX:
    case GridType::DX:
    {
      GridCoordinateFP3D realCoord = getExCoordFP (posAbs);

      /*
       * TODO: add separate for all material grids
       */
      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 0, 0);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 1, 0);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 1, 0);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (1, 1, 1);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 0, 1);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0)) + GridCoordinate3D (0, 1, 1);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0));
      }

      break;
    }
    case GridType::EY:
    case GridType::DY:
    {
      GridCoordinateFP3D realCoord = getEyCoordFP (posAbs);

      /*
       * TODO: add separate for all material grids
       */
      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 1, 0);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 1, 0);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 0, 0);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 1, 1);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 1, 1);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (0, 0, 1);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0)) + GridCoordinate3D (1, 0, 1);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0));
      }

      break;
    }
    case GridType::EZ:
    case GridType::DZ:
    {
      GridCoordinateFP3D realCoord = getEzCoordFP (posAbs);

      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 0, 1);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 1, 1);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 0, 1);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 1, 1);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 0, 0);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (0, 1, 0);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5)) + GridCoordinate3D (1, 1, 0);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5));
        absPos12 = getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5));
      }

      break;
    }
    case GridType::HX:
    case GridType::BX:
    {
      GridCoordinateFP3D realCoord = getHxCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () - 0.5);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () + 0.5);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () - 0.5);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);

        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () + 0.5);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () - 0.5);
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, realCoord.getZ () + 0.5);
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () - 0.5);
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, realCoord.getZ () + 0.5);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HY:
    case GridType::BY:
    {
      GridCoordinateFP3D realCoord = getHyCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () - 0.5);
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), realCoord.getZ () + 0.5);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HZ:
    case GridType::BZ:
    {
      GridCoordinateFP3D realCoord = getHzCoordFP (posAbs);
      GridCoordinateFP3D coord;

      if (isDoubleMaterialPrecision)
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos11 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 0);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos21 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 0);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (1, 0, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos31 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 0);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 1, 1);

        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos41 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 0);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + GridCoordinate3D (0, 0, 1);
      }
      else
      {
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos11 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos12 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, realCoord.getZ ());
        absPos21 = getEpsCoord (coord);
        coord = GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, realCoord.getZ ());
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (typeOfMaterial == GridType::EPS
          || typeOfMaterial == GridType::MU
          || typeOfMaterial == GridType::SIGMAX
          || typeOfMaterial == GridType::SIGMAY
          || typeOfMaterial == GridType::SIGMAZ);

  if (isDoubleMaterialPrecision)
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getApproximateMaterial (gridMaterial, absPos11, absPos12, absPos21, absPos22, absPos31, absPos32, absPos41, absPos42);
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getApproximateMaterial (gridMaterial, absPos11, absPos12);
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return getApproximateMaterial (gridMaterial, absPos11, absPos12, absPos21, absPos22);
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
}

#endif /* YEE_GRID_LAYOUT_H */

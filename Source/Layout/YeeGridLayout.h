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
 * TODO: add link to docs with description of yee layout
 *
 * TODO: consider removing minCoord which is always currently 0
 *
 * TODO: inline getter/setter
 */
template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
class YeeGridLayout
{
public:

  static const bool isParallel;

private:

  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

protected:

  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;

  const TC zeroCoord; /**< Zero coordinate of grid */
  const TC minEpsCoord; /**< Minimum epsilon coordinate */
  const TC minMuCoord; /**< Minimum mu coordinate */
  const TC minExCoord; /**< Minimum Ex field coordinate */
  const TC minEyCoord; /**< Minimum Ey field coordinate */
  const TC minEzCoord; /**< Minimum Ez field coordinate */
  const TC minHxCoord; /**< Minimum Hx field coordinate */
  const TC minHyCoord; /**< Minimum Hy field coordinate */
  const TC minHzCoord; /**< Minimum Hz field coordinate */

  const TCFP zeroCoordFP; /**< Real zero coordinate of grid (corresponding to zeroCoord) */
  const TCFP minEpsCoordFP; /**< Minimum real epsilon coordinate (corresponding to minEpsCoord) */
  const TCFP minMuCoordFP; /**< Minimum real mu coordinate (corresponding to minMuCoord) */
  const TCFP minExCoordFP; /**< Minimum real Ex field coordinate (corresponding to minExCoord) */
  const TCFP minEyCoordFP; /**< Minimum real Ey field coordinate (corresponding to minEyCoord) */
  const TCFP minEzCoordFP; /**< Minimum real Ez field coordinate (corresponding to minEzCoord) */
  const TCFP minHxCoordFP; /**< Minimum real Hx field coordinate (corresponding to minHxCoord) */
  const TCFP minHyCoordFP; /**< Minimum real Hy field coordinate (corresponding to minHyCoord) */
  const TCFP minHzCoordFP; /**< Minimum real Hz field coordinate (corresponding to minHzCoord) */

  const TC size; /**< Size of grids */
  const TC sizeEps; /**< Size of epsilon grid */
  const TC sizeMu; /**< Size of mu grid */
  const TC sizeEx; /**< size of Ex field */
  const TC sizeEy; /**< size of Ey field */
  const TC sizeEz; /**< size of Ez field */
  const TC sizeHx; /**< size of Hx field */
  const TC sizeHy; /**< size of Hy field */
  const TC sizeHz; /**< size of Hz field */

  const TC leftBorderPML; /**< Coordinate of left border of PML */
  const TC rightBorderPML; /**< Coordinate of right border of PML */

  const TC leftBorderTotalField; /**< Coordinate of left border of TF/SF */
  const TC rightBorderTotalField; /**< Coordinate of right border of TF/SF */
  const TCFP leftBorderTotalFieldFP; /**< Floating-point coordinate of left border of TF/SF */
  const TCFP rightBorderTotalFieldFP; /**< Floating-point coordinate of right border of TF/SF */

  const TCFP zeroIncCoordFP; /**< Real coordinate corresponding to zero coordinate of auxiliary grid
                              *   for incident wave */

  const FPValue incidentWaveAngle1; /**< Teta incident wave angle */
  const FPValue incidentWaveAngle2; /**< Phi incident wave angle */
  const FPValue incidentWaveAngle3; /**< Psi incident wave angle */

  const bool isDoubleMaterialPrecision; /**< Flag whether to use double material precision grids */

public:

  /*
   * Get coordinate of circut field component
   */
  TC getExCircuitElement (TC, LayoutDirection) const;
  TC getEyCircuitElement (TC, LayoutDirection) const;
  TC getEzCircuitElement (TC, LayoutDirection) const;
  TC getHxCircuitElement (TC, LayoutDirection) const;
  TC getHyCircuitElement (TC, LayoutDirection) const;
  TC getHzCircuitElement (TC, LayoutDirection) const;

  /*
   * Get size of field component grid
   */
  TC getSize () const
  {
    return size;
  }
  TC getEpsSize () const
  {
    return sizeEps;
  }
  TC getMuSize () const
  {
    return sizeEps;
  }
  TC getExSize () const
  {
    return sizeEx;
  }
  TC getEySize () const
  {
    return sizeEy;
  }
  TC getEzSize () const
  {
    return sizeEz;
  }
  TC getHxSize () const
  {
    return sizeHx;
  }
  TC getHySize () const
  {
    return sizeHy;
  }
  TC getHzSize () const
  {
    return sizeHz;
  }

  TC getSizePML () const
  {
    return leftBorderPML;
  }
  TC getSizeTFSF () const
  {
    return leftBorderTotalField;
  }

  /*
   * Get start coordinate of field component
   */
  TC getExStartDiff () const
  {
    return minExCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  TC getEyStartDiff () const
  {
    return minEyCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  TC getEzStartDiff () const
  {
    return minEzCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  TC getHxStartDiff () const
  {
    return minHxCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  TC getHyStartDiff () const
  {
    return minHyCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  TC getHzStartDiff () const
  {
    return minHzCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }

  /*
   * Get end coordinate of field component
   */
  TC getExEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minExCoord;
  }
  TC getEyEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minEyCoord;
  }
  TC getEzEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minEzCoord;
  }
  TC getHxEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHxCoord;
  }
  TC getHyEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHyCoord;
  }
  TC getHzEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHzCoord;
  }

  /*
   * Get minimum coordinate of field component
   */
  TC getZeroCoord () const
  {
    return zeroCoord;
  }
  TC getMinEpsCoord () const
  {
    return minEpsCoord;
  }
  TC getMinMuCoord () const
  {
    return minMuCoord;
  }
  TC getMinExCoord () const
  {
    return minExCoord;
  }
  TC getMinEyCoord () const
  {
    return minEyCoord;
  }
  TC getMinEzCoord () const
  {
    return minEzCoord;
  }
  TC getMinHxCoord () const
  {
    return minHxCoord;
  }
  TC getMinHyCoord () const
  {
    return minHyCoord;
  }
  TC getMinHzCoord () const
  {
    return minHzCoord;
  }

  /*
   * Get minimum real coordinate of field component
   */
  TCFP getZeroCoordFP () const
  {
    return zeroCoordFP;
  }
  TCFP getMinEpsCoordFP () const
  {
    return minEpsCoordFP;
  }
  TCFP getMinMuCoordFP () const
  {
    return minMuCoordFP;
  }
  TCFP getMinExCoordFP () const
  {
    return minExCoordFP;
  }
  TCFP getMinEyCoordFP () const
  {
    return minEyCoordFP;
  }
  TCFP getMinEzCoordFP () const
  {
    return minEzCoordFP;
  }
  TCFP getMinHxCoordFP () const
  {
    return minHxCoordFP;
  }
  TCFP getMinHyCoordFP () const
  {
    return minHyCoordFP;
  }
  TCFP getMinHzCoordFP () const
  {
    return minHzCoordFP;
  }

  /*
   * Get real coordinate of field component by its coordinate
   */
  TCFP getEpsCoordFP (TC coord) const
  {
    return convertCoord (coord - minEpsCoord) + minEpsCoordFP;
  }
  TCFP getMuCoordFP (TC coord) const
  {
    return convertCoord (coord - minMuCoord) + minMuCoordFP;
  }
  TCFP getExCoordFP (TC coord) const
  {
    return convertCoord (coord - minExCoord) + minExCoordFP;
  }
  TCFP getEyCoordFP (TC coord) const
  {
    return convertCoord (coord - minEyCoord) + minEyCoordFP;
  }
  TCFP getEzCoordFP (TC coord) const
  {
    return convertCoord (coord - minEzCoord) + minEzCoordFP;
  }
  TCFP getHxCoordFP (TC coord) const
  {
    return convertCoord (coord - minHxCoord) + minHxCoordFP;
  }
  TCFP getHyCoordFP (TC coord) const
  {
    return convertCoord (coord - minHyCoord) + minHyCoordFP;
  }
  TCFP getHzCoordFP (TC coord) const
  {
    return convertCoord (coord - minHzCoord) + minHzCoordFP;
  }

  /*
   * Get coordinate of field component by its real coordinate
   */
  TC getEpsCoord (TCFP coord) const
  {
    return convertCoord (coord - minEpsCoordFP) + minEpsCoord;
  }
  TC getMuCoord (TCFP coord) const
  {
    return convertCoord (coord - minMuCoordFP) + minMuCoord;
  }
  TC getExCoord (TCFP coord) const
  {
    return convertCoord (coord - minExCoordFP) + minExCoord;
  }
  TC getEyCoord (TCFP coord) const
  {
    return convertCoord (coord - minEyCoordFP) + minEyCoord;
  }
  TC getEzCoord (TCFP coord) const
  {
    return convertCoord (coord - minEzCoordFP) + minEzCoord;
  }
  TC getHxCoord (TCFP coord) const
  {
    return convertCoord (coord - minHxCoordFP) + minHxCoord;
  }
  TC getHyCoord (TCFP coord) const
  {
    return convertCoord (coord - minHyCoordFP) + minHyCoord;
  }
  TC getHzCoord (TCFP coord) const
  {
    return convertCoord (coord - minHzCoordFP) + minHzCoord;
  }

  /*
   * PML
   */
  TC getLeftBorderPML () const
  {
    return leftBorderPML;
  }
  TC getRightBorderPML () const
  {
    return rightBorderPML;
  }

  bool isExInPML (TC coord) const
  {
    return isInPML (getExCoordFP (coord));
  }
  bool isEyInPML (TC coord) const
  {
    return isInPML (getEyCoordFP (coord));
  }
  bool isEzInPML (TC coord) const
  {
    return isInPML (getEzCoordFP (coord));
  }
  bool isHxInPML (TC coord) const
  {
    return isInPML (getHxCoordFP (coord));
  }
  bool isHyInPML (TC coord) const
  {
    return isInPML (getHyCoordFP (coord));
  }
  bool isHzInPML (TC coord) const
  {
    return isInPML (getHzCoordFP (coord));
  }

  /*
   * Total field / scattered field
   */
  TC getLeftBorderTFSF () const
  {
    return leftBorderTotalField;
  }
  TC getRightBorderTFSF () const
  {
    return rightBorderTotalField;
  }
  TCFP getZeroIncCoordFP () const
  {
    return zeroIncCoordFP;
  }
  bool doNeedTFSFUpdateExBorder (TC, LayoutDirection) const;
  bool doNeedTFSFUpdateEyBorder (TC, LayoutDirection) const;
  bool doNeedTFSFUpdateEzBorder (TC, LayoutDirection) const;
  bool doNeedTFSFUpdateHxBorder (TC, LayoutDirection) const;
  bool doNeedTFSFUpdateHyBorder (TC, LayoutDirection) const;
  bool doNeedTFSFUpdateHzBorder (TC, LayoutDirection) const;

  FPValue getIncidentWaveAngle1 () const
  {
    return incidentWaveAngle1;
  }
  FPValue getIncidentWaveAngle2 () const
  {
    return incidentWaveAngle2;
  }
  FPValue getIncidentWaveAngle3 () const
  {
    return incidentWaveAngle3;
  }

  bool getIsDoubleMaterialPrecision () const
  {
    return isDoubleMaterialPrecision;
  }

  FieldValue getExFromIncidentE (FieldValue valE) const
  {
    return valE * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
  }
  FieldValue getEyFromIncidentE (FieldValue valE) const
  {
    return valE * (FPValue) ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
  }
  FieldValue getEzFromIncidentE (FieldValue valE) const
  {
    return valE * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle1));
  }
  FieldValue getHxFromIncidentH (FieldValue valH) const
  {
    return valH * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
  }
  FieldValue getHyFromIncidentH (FieldValue valH) const
  {
    return valH * (FPValue) (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
  }
  FieldValue getHzFromIncidentH (FieldValue valH) const
  {
    return - valH * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle1));
  }

  /**
   * Constructor of Yee grid
   */
  YeeGridLayout (TC coordSize,
                 TC sizePML,
                 TC sizeScatteredZone,
                 FPValue incWaveAngle1, /**< teta */
                 FPValue incWaveAngle2, /**< phi */
                 FPValue incWaveAngle3, /**< psi */
                 bool doubleMaterialPrecision);

  ~YeeGridLayout ()
  {
  } /* ~YeeGridLayout */

  FPValue getApproximateMaterial (Grid<TC> *, TC, TC);
  FPValue getApproximateMaterial (Grid<TC> *, TC, TC, TC, TC);
  FPValue getApproximateMaterial (Grid<TC> *, TC, TC, TC, TC, TC, TC, TC, TC);
  FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, FPValue &, FPValue &);
  FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, TC, TC, FPValue &, FPValue &);
  FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, TC, TC, TC, TC, TC, TC,
                                      FPValue &, FPValue &);
  FPValue getMetaMaterial (const TC &, GridType, Grid<TC> *, GridType, Grid<TC> *, GridType, Grid<TC> *, GridType,
                           FPValue &, FPValue &);
  FPValue getMaterial (const TC &, GridType, Grid<TC> *, GridType);

  template <bool isMetaMaterial>
  void initCoordinates (const TC &, GridType, TC &, TC &, TC &, TC &, TC &, TC &, TC &, TC &);

private:

  bool isInPML (TCFP realCoordFP) const;
}; /* YeeGridLayout */

class YeeGridLayoutHelper
{
  template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
  friend class YeeGridLayout;

private:

  template<int8_t Offset>
  static bool tfsfBorder (FPValue coordFP, FPValue borderFP)
  {
    ASSERT (Offset > -10 && Offset < 10);
    return IS_FP_EXACT (coordFP, borderFP + Offset * FPValue (1) / FPValue (10));
  }

  template<int8_t Offset1, int8_t Offset2>
  static bool tfsfBorder (FPValue coordFP, FPValue firstBorderFP, FPValue secondBorderFP)
  {
    ASSERT (Offset1 > -10 && Offset1 < 10);
    ASSERT (Offset2 > -10 && Offset2 < 10);
    return coordFP > firstBorderFP + Offset1 * FPValue (1) / FPValue (10)
           && coordFP < secondBorderFP + Offset2 * FPValue (1) / FPValue (10);
  }

  template<int8_t T>
  static bool tfsfBorder1DFirst__1 (GridCoordinateFP1D coord, GridCoordinateFP1D leftBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T> (coord.get1 (), leftBorderTotalFieldFP.get1 ());
  }
  template<int8_t T>
  static bool tfsfBorder1DSecond__1 (GridCoordinateFP1D coord, GridCoordinateFP1D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T> (coord.get1 (), rightBorderTotalFieldFP.get1 ());
  }
  template<int8_t T1, int8_t T2>
  static bool tfsfBorder2DFirst__1 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get1 (), leftBorderTotalFieldFP.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T2> (coord.get2 (), leftBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  static bool tfsfBorder2DSecond__1 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get1 (), leftBorderTotalFieldFP.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T2> (coord.get2 (), rightBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  static bool tfsfBorder2DFirst__2 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T2> (coord.get1 (), leftBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get2 (), leftBorderTotalFieldFP.get2 (), rightBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  static bool tfsfBorder2DSecond__2 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T2> (coord.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get2 (), leftBorderTotalFieldFP.get2 (), rightBorderTotalFieldFP.get2 ());
  }

  static bool doNeedTFSFUpdateBorder (LayoutDirection dir,
                                      bool condL,
                                      bool condR,
                                      bool condD,
                                      bool condU,
                                      bool condB,
                                      bool condF)
  {
    bool match = false;

    switch (dir)
    {
      case LayoutDirection::LEFT:
      {
        match = condL;
        break;
      }
      case LayoutDirection::RIGHT:
      {
        match = condR;
        break;
      }
      case LayoutDirection::DOWN:
      {
        match = condD;
        break;
      }
      case LayoutDirection::UP:
      {
        match = condU;
        break;
      }
      case LayoutDirection::BACK:
      {
        match = condB;
        break;
      }
      case LayoutDirection::FRONT:
      {
        match = condF;
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return match;
  }

  static bool isInPML1D (GridCoordinateFP1D realCoordFP, GridCoordinateFP1D zeroCoordFP,
                         GridCoordinate1D leftBorderPML, GridCoordinate1D rightBorderPML)
  {
    GridCoordinateFP1D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
    GridCoordinateFP1D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

    /*
     * TODO: remove floating point equality comparison
     */
    ASSERT (coordLeftBorderPMLFP < coordRightBorderPMLFP);
    bool isInXPML = realCoordFP.get1 () < coordLeftBorderPMLFP.get1 ()
                    || realCoordFP.get1 () >= coordRightBorderPMLFP.get1 ();

    return isInXPML;
  }

  static bool isInPML2D (GridCoordinateFP2D realCoordFP, GridCoordinateFP2D zeroCoordFP,
                         GridCoordinate2D leftBorderPML, GridCoordinate2D rightBorderPML)
  {
    GridCoordinateFP2D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
    GridCoordinateFP2D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

    /*
     * TODO: remove floating point equality comparison
     */
    ASSERT (coordLeftBorderPMLFP < coordRightBorderPMLFP);
    bool isInXPML = realCoordFP.get1 () < coordLeftBorderPMLFP.get1 ()
                    || realCoordFP.get1 () >= coordRightBorderPMLFP.get1 ();
    bool isInYPML = realCoordFP.get2 () < coordLeftBorderPMLFP.get2 ()
                    || realCoordFP.get2 () >= coordRightBorderPMLFP.get2 ();

    return isInXPML || isInYPML;
  }

  static bool isInPML3D (GridCoordinateFP3D realCoordFP, GridCoordinateFP3D zeroCoordFP,
                         GridCoordinate3D leftBorderPML, GridCoordinate3D rightBorderPML)
  {
    GridCoordinateFP3D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
    GridCoordinateFP3D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

    /*
     * TODO: remove floating point equality comparison
     */
    ASSERT (coordLeftBorderPMLFP < coordRightBorderPMLFP);
    bool isInXPML = realCoordFP.get1 () < coordLeftBorderPMLFP.get1 ()
                    || realCoordFP.get1 () >= coordRightBorderPMLFP.get1 ();
    bool isInYPML = realCoordFP.get2 () < coordLeftBorderPMLFP.get2 ()
                    || realCoordFP.get2 () >= coordRightBorderPMLFP.get2 ();
    bool isInZPML = realCoordFP.get3 () < coordLeftBorderPMLFP.get3 ()
                    || realCoordFP.get3 () >= coordRightBorderPMLFP.get3 ();

    return isInXPML || isInYPML || isInZPML;
  }
};

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMaterial (Grid<YeeGridLayout::TC> *gridMaterial,
                                       TC coord1,
                                       TC coord2)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);

  return Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                             Approximation::getMaterial (val2));
}

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMaterial (Grid<TC> *gridMaterial,
                                       TC coord1,
                                       TC coord2,
                                       TC coord3,
                                       TC coord4)
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMaterial (Grid<TC> *gridMaterial,
                                       TC coord1,
                                       TC coord2,
                                       TC coord3,
                                       TC coord4,
                                       TC coord5,
                                       TC coord6,
                                       TC coord7,
                                       TC coord8)
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMetaMaterial (Grid<TC> *gridMaterial,
                                           Grid<TC> *gridMaterialOmega,
                                           Grid<TC> *gridMaterialGamma,
                                           TC coord1,
                                           TC coord2,
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMetaMaterial (Grid<TC> *gridMaterial,
                                           Grid<TC> *gridMaterialOmega,
                                           Grid<TC> *gridMaterialGamma,
                                           TC coord1,
                                           TC coord2,
                                           TC coord3,
                                           TC coord4,
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMetaMaterial (Grid<TC> *gridMaterial,
                                           Grid<TC> *gridMaterialOmega,
                                           Grid<TC> *gridMaterialGamma,
                                           TC coord1,
                                           TC coord2,
                                           TC coord3,
                                           TC coord4,
                                           TC coord5,
                                           TC coord6,
                                           TC coord7,
                                           TC coord8,
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getMetaMaterial (const TC &posAbs,
                                GridType typeOfField,
                                Grid<TC> *gridMaterial,
                                GridType typeOfMaterial,
                                Grid<TC> *gridMaterialOmega,
                                GridType typeOfMaterialOmega,
                                Grid<TC> *gridMaterialGamma,
                                GridType typeOfMaterialGamma,
                                FPValue &omega,
                                FPValue &gamma)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  initCoordinates<true> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                         absPos31, absPos32, absPos41, absPos42);

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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
FPValue
YeeGridLayout<Type, TCoord, layout_type>::getMaterial (const TC &posAbs,
                            GridType typeOfField,
                            Grid<TC> *gridMaterial,
                            GridType typeOfMaterial)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  initCoordinates<false> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                          absPos31, absPos32, absPos41, absPos42);

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

/*
 * TODO: add correct approximation for metamaterials
 *
 * TODO: add separate for all fields
 */
template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
template <bool isMetaMaterial>
void
YeeGridLayout<Type, TCoord, layout_type>::initCoordinates (const TC &posAbs,
                                     GridType typeOfField,
                                     TC &absPos11,
                                     TC &absPos12,
                                     TC &absPos21,
                                     TC &absPos22,
                                     TC &absPos31,
                                     TC &absPos32,
                                     TC &absPos41,
                                     TC &absPos42)
{
  switch (typeOfField)
  {
    case GridType::EX:
    case GridType::DX:
    {
      TCFP realCoord = getExCoordFP (posAbs);

      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3));
        absPos12 = getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3));
      }

      break;
    }
    case GridType::EY:
    case GridType::DY:
    {
      TCFP realCoord = getEyCoordFP (posAbs);

      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3));
        absPos12 = getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3));
      }

      break;
    }
    case GridType::EZ:
    case GridType::DZ:
    {
      TCFP realCoord = getEzCoordFP (posAbs);

      if (isDoubleMaterialPrecision)
      {
        absPos11 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);

        absPos21 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        absPos31 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);

        absPos41 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);
      }
      else
      {
        absPos11 = getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3));
        absPos12 = getEpsCoord (realCoord + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3));
      }

      break;
    }
    case GridType::HX:
    case GridType::BX:
    {
      TCFP realCoord = getHxCoordFP (posAbs);
      TCFP coord;

      if (isDoubleMaterialPrecision)
      {
        coord = realCoord + TCSFP::initAxesCoordinate (0.0, -0.5, -0.5, ct1, ct2, ct3);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.0, -0.5, 0.5, ct1, ct2, ct3);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.0, 0.5, -0.5, ct1, ct2, ct3);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.0, 0.5, 0.5, ct1, ct2, ct3);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
      }
      else
      {
        coord = realCoord + TCSFP::initAxesCoordinate (0.0, -0.5, -0.5, ct1, ct2, ct3);
        absPos11 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.0, -0.5, 0.5, ct1, ct2, ct3);
        absPos12 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.0, 0.5, -0.5, ct1, ct2, ct3);
        absPos21 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.0, 0.5, 0.5, ct1, ct2, ct3);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HY:
    case GridType::BY:
    {
      TCFP realCoord = getHyCoordFP (posAbs);
      TCFP coord;

      if (isDoubleMaterialPrecision)
      {
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.0, -0.5, ct1, ct2, ct3);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.0, 0.5, ct1, ct2, ct3);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.0, -0.5, ct1, ct2, ct3);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.0, 0.5, ct1, ct2, ct3);
        absPos41 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
      }
      else
      {
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.0, -0.5, ct1, ct2, ct3);
        absPos11 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.0, 0.5, ct1, ct2, ct3);
        absPos12 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.0, -0.5, ct1, ct2, ct3);
        absPos21 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.0, 0.5, ct1, ct2, ct3);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    case GridType::HZ:
    case GridType::BZ:
    {
      TCFP realCoord = getHzCoordFP (posAbs);
      TCFP coord;

      if (isDoubleMaterialPrecision)
      {
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, -0.5, 0.0, ct1, ct2, ct3);
        absPos11 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);
        absPos12 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.5, 0.0, ct1, ct2, ct3);
        absPos21 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
        absPos22 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (1, 0, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.5, -0.5, 0.0, ct1, ct2, ct3);
        absPos31 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
        absPos32 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 1, 1, ct1, ct2, ct3);

        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.5, 0.0), ct1, ct2, ct3;
        absPos41 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
        absPos42 = grid_coord (2) * getEpsCoord (coord) + TC::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
      }
      else
      {
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, -0.5, 0.0, ct1, ct2, ct3);
        absPos11 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (-0.5, 0.5, 0.0, ct1, ct2, ct3);
        absPos12 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.5, -0.5, 0.0, ct1, ct2, ct3);
        absPos21 = getEpsCoord (coord);
        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.5, 0.0, ct1, ct2, ct3);
        absPos22 = getEpsCoord (coord);
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getExCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minExCoord) + minExCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getEyCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minEyCoord) + minEyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getEzCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minEzCoord) + minEzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minHxCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getHxCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minHxCoord) + minHxCoordFP;

  switch (dir)
  {
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getHyCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minHyCoord) + minHyCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEzCoordFP;
      break;
    }
    case LayoutDirection::BACK:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::FRONT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0, 0.5, ct1, ct2, ct3);
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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
typename YeeGridLayout<Type, TCoord, layout_type>::TC
YeeGridLayout<Type, TCoord, layout_type>::getHzCircuitElement (TC coord, LayoutDirection dir) const
{
  TCFP coordFP = convertCoord (coord - minHzCoord) + minHzCoordFP;

  switch (dir)
  {
    case LayoutDirection::LEFT:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::RIGHT:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3);
      coordFP = coordFP - minEyCoordFP;
      break;
    }
    case LayoutDirection::DOWN:
    {
      coordFP = coordFP - TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
      coordFP = coordFP - minExCoordFP;
      break;
    }
    case LayoutDirection::UP:
    {
      coordFP = coordFP + TCSFP::initAxesCoordinate (0, 0.5, 0, ct1, ct2, ct3);
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

#endif /* YEE_GRID_LAYOUT_H */

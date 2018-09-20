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
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
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
  CUDA_DEVICE CUDA_HOST TC getExCircuitElement (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST TC getEyCircuitElement (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST TC getEzCircuitElement (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST TC getHxCircuitElement (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST TC getHyCircuitElement (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST TC getHzCircuitElement (TC, LayoutDirection) const;

  /*
   * Get size of field component grid
   */
  CUDA_DEVICE CUDA_HOST TC getSize () const
  {
    return size;
  }
  CUDA_DEVICE CUDA_HOST TC getEpsSize () const
  {
    return sizeEps;
  }
  CUDA_DEVICE CUDA_HOST TC getMuSize () const
  {
    return sizeEps;
  }
  CUDA_DEVICE CUDA_HOST TC getExSize () const
  {
    return sizeEx;
  }
  CUDA_DEVICE CUDA_HOST TC getEySize () const
  {
    return sizeEy;
  }
  CUDA_DEVICE CUDA_HOST TC getEzSize () const
  {
    return sizeEz;
  }
  CUDA_DEVICE CUDA_HOST TC getHxSize () const
  {
    return sizeHx;
  }
  CUDA_DEVICE CUDA_HOST TC getHySize () const
  {
    return sizeHy;
  }
  CUDA_DEVICE CUDA_HOST TC getHzSize () const
  {
    return sizeHz;
  }

  CUDA_DEVICE CUDA_HOST TC getSizePML () const
  {
    return leftBorderPML;
  }

  /*
   * Get start coordinate of field component
   */
  CUDA_DEVICE CUDA_HOST TC getExStartDiff () const
  {
    return minExCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  CUDA_DEVICE CUDA_HOST TC getEyStartDiff () const
  {
    return minEyCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  CUDA_DEVICE CUDA_HOST TC getEzStartDiff () const
  {
    return minEzCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  CUDA_DEVICE CUDA_HOST TC getHxStartDiff () const
  {
    return minHxCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  CUDA_DEVICE CUDA_HOST TC getHyStartDiff () const
  {
    return minHyCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }
  CUDA_DEVICE CUDA_HOST TC getHzStartDiff () const
  {
    return minHzCoord + TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
  }

  /*
   * Get end coordinate of field component
   */
  CUDA_DEVICE CUDA_HOST TC getExEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minExCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getEyEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minEyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getEzEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minEzCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHxEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHxCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHyEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHzEndDiff () const
  {
    return TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3) - minHzCoord;
  }

  /*
   * Get minimum coordinate of field component
   */
  CUDA_DEVICE CUDA_HOST TC getZeroCoord () const
  {
    return zeroCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinEpsCoord () const
  {
    return minEpsCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinMuCoord () const
  {
    return minMuCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinExCoord () const
  {
    return minExCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinEyCoord () const
  {
    return minEyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinEzCoord () const
  {
    return minEzCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinHxCoord () const
  {
    return minHxCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinHyCoord () const
  {
    return minHyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMinHzCoord () const
  {
    return minHzCoord;
  }

  /*
   * Get minimum real coordinate of field component
   */
  CUDA_DEVICE CUDA_HOST TCFP getZeroCoordFP () const
  {
    return zeroCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinEpsCoordFP () const
  {
    return minEpsCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinMuCoordFP () const
  {
    return minMuCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinExCoordFP () const
  {
    return minExCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinEyCoordFP () const
  {
    return minEyCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinEzCoordFP () const
  {
    return minEzCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinHxCoordFP () const
  {
    return minHxCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinHyCoordFP () const
  {
    return minHyCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMinHzCoordFP () const
  {
    return minHzCoordFP;
  }

  /*
   * Get real coordinate of field component by its coordinate
   */
  CUDA_DEVICE CUDA_HOST TCFP getEpsCoordFP (TC coord) const
  {
    return convertCoord (coord - minEpsCoord) + minEpsCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getMuCoordFP (TC coord) const
  {
    return convertCoord (coord - minMuCoord) + minMuCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getExCoordFP (TC coord) const
  {
    return convertCoord (coord - minExCoord) + minExCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getEyCoordFP (TC coord) const
  {
    return convertCoord (coord - minEyCoord) + minEyCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getEzCoordFP (TC coord) const
  {
    return convertCoord (coord - minEzCoord) + minEzCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getHxCoordFP (TC coord) const
  {
    return convertCoord (coord - minHxCoord) + minHxCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getHyCoordFP (TC coord) const
  {
    return convertCoord (coord - minHyCoord) + minHyCoordFP;
  }
  CUDA_DEVICE CUDA_HOST TCFP getHzCoordFP (TC coord) const
  {
    return convertCoord (coord - minHzCoord) + minHzCoordFP;
  }

  /*
   * Get coordinate of field component by its real coordinate
   */
  CUDA_DEVICE CUDA_HOST TC getEpsCoord (TCFP coord) const
  {
    return convertCoord (coord - minEpsCoordFP) + minEpsCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getMuCoord (TCFP coord) const
  {
    return convertCoord (coord - minMuCoordFP) + minMuCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getExCoord (TCFP coord) const
  {
    return convertCoord (coord - minExCoordFP) + minExCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getEyCoord (TCFP coord) const
  {
    return convertCoord (coord - minEyCoordFP) + minEyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getEzCoord (TCFP coord) const
  {
    return convertCoord (coord - minEzCoordFP) + minEzCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHxCoord (TCFP coord) const
  {
    return convertCoord (coord - minHxCoordFP) + minHxCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHyCoord (TCFP coord) const
  {
    return convertCoord (coord - minHyCoordFP) + minHyCoord;
  }
  CUDA_DEVICE CUDA_HOST TC getHzCoord (TCFP coord) const
  {
    return convertCoord (coord - minHzCoordFP) + minHzCoord;
  }

  /*
   * PML
   */
  CUDA_DEVICE CUDA_HOST TC getLeftBorderPML () const
  {
    return leftBorderPML;
  }
  CUDA_DEVICE CUDA_HOST TC getRightBorderPML () const
  {
    return rightBorderPML;
  }

  CUDA_DEVICE CUDA_HOST bool isExInPML (TC coord) const
  {
    return isInPML (getExCoordFP (coord));
  }
  CUDA_DEVICE CUDA_HOST bool isEyInPML (TC coord) const
  {
    return isInPML (getEyCoordFP (coord));
  }
  CUDA_DEVICE CUDA_HOST bool isEzInPML (TC coord) const
  {
    return isInPML (getEzCoordFP (coord));
  }
  CUDA_DEVICE CUDA_HOST bool isHxInPML (TC coord) const
  {
    return isInPML (getHxCoordFP (coord));
  }
  CUDA_DEVICE CUDA_HOST bool isHyInPML (TC coord) const
  {
    return isInPML (getHyCoordFP (coord));
  }
  CUDA_DEVICE CUDA_HOST bool isHzInPML (TC coord) const
  {
    return isInPML (getHzCoordFP (coord));
  }

  /*
   * Total field / scattered field
   */
  CUDA_DEVICE CUDA_HOST TC getLeftBorderTFSF () const
  {
    return leftBorderTotalField;
  }
  CUDA_DEVICE CUDA_HOST TC getRightBorderTFSF () const
  {
    return rightBorderTotalField;
  }
  CUDA_DEVICE CUDA_HOST TCFP getZeroIncCoordFP () const
  {
    return zeroIncCoordFP;
  }
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateExBorder (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateEyBorder (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateEzBorder (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateHxBorder (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateHyBorder (TC, LayoutDirection) const;
  CUDA_DEVICE CUDA_HOST bool doNeedTFSFUpdateHzBorder (TC, LayoutDirection) const;

  CUDA_DEVICE CUDA_HOST FPValue getIncidentWaveAngle1 () const
  {
    return incidentWaveAngle1;
  }
  CUDA_DEVICE CUDA_HOST FPValue getIncidentWaveAngle2 () const
  {
    return incidentWaveAngle2;
  }
  CUDA_DEVICE CUDA_HOST FPValue getIncidentWaveAngle3 () const
  {
    return incidentWaveAngle3;
  }

  CUDA_DEVICE CUDA_HOST bool getIsDoubleMaterialPrecision () const
  {
    return isDoubleMaterialPrecision;
  }

  CUDA_DEVICE CUDA_HOST FieldValue getExFromIncidentE (FieldValue valE) const;
  CUDA_DEVICE CUDA_HOST FieldValue getEyFromIncidentE (FieldValue valE) const;
  CUDA_DEVICE CUDA_HOST FieldValue getEzFromIncidentE (FieldValue valE) const;
  CUDA_DEVICE CUDA_HOST FieldValue getHxFromIncidentH (FieldValue valH) const;
  CUDA_DEVICE CUDA_HOST FieldValue getHyFromIncidentH (FieldValue valH) const;
  CUDA_DEVICE CUDA_HOST FieldValue getHzFromIncidentH (FieldValue valH) const;

  /**
   * Constructor of Yee grid
   */
  CUDA_DEVICE CUDA_HOST YeeGridLayout (TC coordSize,
                 TC sizePML,
                 TC sizeScatteredZoneLeft,
                 TC sizeScatteredZoneRight,
                 FPValue incWaveAngle1, /**< teta */
                 FPValue incWaveAngle2, /**< phi */
                 FPValue incWaveAngle3, /**< psi */
                 bool doubleMaterialPrecision);

  CUDA_DEVICE CUDA_HOST ~YeeGridLayout ()
  {
  } /* ~YeeGridLayout */

  CUDA_DEVICE CUDA_HOST FPValue getApproximateMaterial (Grid<TC> *, TC, TC);
  CUDA_DEVICE CUDA_HOST FPValue getApproximateMaterial (Grid<TC> *, TC, TC, TC, TC);
  CUDA_DEVICE CUDA_HOST FPValue getApproximateMaterial (Grid<TC> *, TC, TC, TC, TC, TC, TC, TC, TC);
  CUDA_DEVICE CUDA_HOST FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, FPValue &, FPValue &);
  CUDA_DEVICE CUDA_HOST FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, TC, TC, FPValue &, FPValue &);
  CUDA_DEVICE CUDA_HOST FPValue getApproximateMetaMaterial (Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC, TC, TC, TC, TC, TC, TC,
                                      FPValue &, FPValue &);
  CUDA_DEVICE CUDA_HOST FPValue getMetaMaterial (const TC &, GridType, Grid<TC> *, GridType, Grid<TC> *, GridType, Grid<TC> *, GridType,
                           FPValue &, FPValue &);
  CUDA_DEVICE CUDA_HOST FPValue getMaterial (const TC &, GridType, Grid<TC> *, GridType);

  template <bool isMetaMaterial>
  CUDA_DEVICE CUDA_HOST void initCoordinates (const TC &, GridType, TC &, TC &, TC &, TC &, TC &, TC &, TC &, TC &);

private:

  CUDA_DEVICE CUDA_HOST bool isInPML (TCFP realCoordFP) const;
}; /* YeeGridLayout */

class YeeGridLayoutHelper
{
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  friend class YeeGridLayout;

private:

  template<int8_t Offset>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder (FPValue coordFP, FPValue borderFP)
  {
    ASSERT (Offset > -10 && Offset < 10);
    return IS_FP_EXACT (coordFP, borderFP + Offset * FPValue (1) / FPValue (10));
  }

  template<int8_t Offset1, int8_t Offset2>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder (FPValue coordFP, FPValue firstBorderFP, FPValue secondBorderFP)
  {
    ASSERT (Offset1 > -10 && Offset1 < 10);
    ASSERT (Offset2 > -10 && Offset2 < 10);
    return coordFP > firstBorderFP + Offset1 * FPValue (1) / FPValue (10)
           && coordFP < secondBorderFP + Offset2 * FPValue (1) / FPValue (10);
  }

  template<int8_t T>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder1DFirst__1 (GridCoordinateFP1D coord, GridCoordinateFP1D leftBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T> (coord.get1 (), leftBorderTotalFieldFP.get1 ());
  }
  template<int8_t T>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder1DSecond__1 (GridCoordinateFP1D coord, GridCoordinateFP1D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T> (coord.get1 (), rightBorderTotalFieldFP.get1 ());
  }
  template<int8_t T1, int8_t T2>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder2DFirst__1 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get1 (), leftBorderTotalFieldFP.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T2> (coord.get2 (), leftBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder2DSecond__1 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get1 (), leftBorderTotalFieldFP.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T2> (coord.get2 (), rightBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder2DFirst__2 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T2> (coord.get1 (), leftBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get2 (), leftBorderTotalFieldFP.get2 (), rightBorderTotalFieldFP.get2 ());
  }
  template<int8_t T1, int8_t T2>
  CUDA_DEVICE CUDA_HOST static bool tfsfBorder2DSecond__2 (GridCoordinateFP2D coord, GridCoordinateFP2D leftBorderTotalFieldFP, GridCoordinateFP2D rightBorderTotalFieldFP)
  {
    return YeeGridLayoutHelper::tfsfBorder<T2> (coord.get1 (), rightBorderTotalFieldFP.get1 ())
           && YeeGridLayoutHelper::tfsfBorder<T1, -T1> (coord.get2 (), leftBorderTotalFieldFP.get2 (), rightBorderTotalFieldFP.get2 ());
  }

  CUDA_DEVICE CUDA_HOST static bool doNeedTFSFUpdateBorder (LayoutDirection dir,
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

  CUDA_DEVICE CUDA_HOST static bool isInPML1D (GridCoordinateFP1D realCoordFP, GridCoordinateFP1D zeroCoordFP,
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

  CUDA_DEVICE CUDA_HOST static bool isInPML2D (GridCoordinateFP2D realCoordFP, GridCoordinateFP2D zeroCoordFP,
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

  CUDA_DEVICE CUDA_HOST static bool isInPML3D (GridCoordinateFP3D realCoordFP, GridCoordinateFP3D zeroCoordFP,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
YeeGridLayout<Type, TCoord, layout_type>::getApproximateMaterial (Grid<YeeGridLayout::TC> *gridMaterial,
                                       TC coord1,
                                       TC coord2)
{
  FieldPointValue* val1 = gridMaterial->getFieldPointValueByAbsolutePos (coord1);
  FieldPointValue* val2 = gridMaterial->getFieldPointValueByAbsolutePos (coord2);

  return Approximation::approximateMaterial (Approximation::getMaterial (val1),
                                             Approximation::getMaterial (val2));
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST FPValue
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
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool isMetaMaterial>
CUDA_DEVICE CUDA_HOST void
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
        absPos12 = grid_coord (2) * getEpsCoord (realCoord - TCSFP::initAxesCoordinate (0.5, 0, 0, ct1, ct2, ct3)) + TC::initAxesCoordinate (1, 1, 0, ct1, ct2, ct3);

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

        coord = realCoord + TCSFP::initAxesCoordinate (0.5, 0.5, 0.0, ct1, ct2, ct3);
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_DEVICE CUDA_HOST typename YeeGridLayout<Type, TCoord, layout_type>::TC
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

typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_ExHy;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_ExHz;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_EyHx;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_EyHz;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_EzHx;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED> YL1D_Dim1_EzHy;

typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TEx;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TEy;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TEz;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TMx;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TMy;
typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED> YL2D_Dim2_TMz;

typedef YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED> YL3D_Dim3;

/* Dim1_ExHy */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return - valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return - valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim1_ExHz */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH;
}

/* Dim1_EyHx */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return - valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim1_EyHz */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return - valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH;
}

/* Dim1_EzHx */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim1_EzHy */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return - valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim2_TEx */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) ( - cos (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (sin (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim2_TEy */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (- cos (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (sin (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return - valH;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim2_TEz */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (sin (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) ( - cos (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH;
}

/* Dim2_TMx */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (cos (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH * (FPValue) (sin (incidentWaveAngle1));
}

/* Dim2_TMy */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return - valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (cos (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH * (FPValue) (sin (incidentWaveAngle1));
}

/* Dim2_TMz */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  UNREACHABLE;
  return FPValue (0);
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE;
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (sin (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (- cos (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  UNREACHABLE;
  return FPValue (0);
}

/* Dim3 */
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getExFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getEyFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getEzFromIncidentE (FieldValue valE) const
{
  return valE * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle1));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getHxFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getHyFromIncidentH (FieldValue valH) const
{
  return valH * (FPValue) (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
}
template <>
CUDA_DEVICE CUDA_HOST inline FieldValue
YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::getHzFromIncidentH (FieldValue valH) const
{
  return - valH * (FPValue) (cos (incidentWaveAngle3) * sin (incidentWaveAngle1));
}

#endif /* YEE_GRID_LAYOUT_H */

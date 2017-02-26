#ifndef YEE_GRID_LAYOUT_H
#define YEE_GRID_LAYOUT_H

#include "Assert.h"
#include "GridLayout.h"
#include "PhysicsConst.h"

#include <cmath>

/**
 * Yee grid layout which specifies how field components are placed in space
 *
 * FIXME: add link to docs with description of yee layout
 *
 * FIXME: consider removing minCoord which is always currently 0
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
  virtual GridCoordinate3D getEpsSize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMuSize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getExSize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEySize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEzSize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHxSize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHySize () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHzSize () const CXX11_OVERRIDE_FINAL;

  virtual GridCoordinate3D getSizePML () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getSizeTFSF () const CXX11_OVERRIDE_FINAL;

  /*
   * Get start coordinate of field component
   */
  virtual GridCoordinate3D getExStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEyStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEzStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHxStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHyStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHzStart (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;

  /*
   * Get end coordinate of field component
   */
  virtual GridCoordinate3D getExEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEyEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEzEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHxEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHyEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHzEnd (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;

  /*
   * Get minimum coordinate of field component
   */
  virtual GridCoordinate3D getZeroCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinEpsCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinMuCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinExCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinEyCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinEzCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinHxCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinHyCoord () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMinHzCoord () const CXX11_OVERRIDE_FINAL;

  /*
   * Get minimum real coordinate of field component
   */
  virtual GridCoordinateFP3D getZeroCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinEpsCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinMuCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinExCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinEyCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinEzCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinHxCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinHyCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMinHzCoordFP () const CXX11_OVERRIDE_FINAL;

  /*
   * Get real coordinate of field component by its coordinate
   */
  virtual GridCoordinateFP3D getEpsCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getMuCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getExCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getEyCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getEzCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getHxCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getHyCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getHzCoordFP (GridCoordinate3D) const CXX11_OVERRIDE_FINAL;

  /*
   * Get coordinate of field component by its real coordinate
   */
  virtual GridCoordinate3D getEpsCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getMuCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getExCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEyCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getEzCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHxCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHyCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getHzCoord (GridCoordinateFP3D) const CXX11_OVERRIDE_FINAL;

  /*
   * PML
   */
  virtual GridCoordinate3D getLeftBorderPML () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getRightBorderPML () const CXX11_OVERRIDE_FINAL;
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
  virtual GridCoordinate3D getLeftBorderTFSF () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinate3D getRightBorderTFSF () const CXX11_OVERRIDE_FINAL;
  virtual GridCoordinateFP3D getZeroIncCoordFP () const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateExBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateEyBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateEzBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHxBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHyBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;
  virtual bool doNeedTFSFUpdateHzBorder (GridCoordinate3D, LayoutDirection, bool) const CXX11_OVERRIDE_FINAL;

  /**
   * Constructor of Yee grid
   */
  YeeGridLayout (GridCoordinate3D coordSize,
                 GridCoordinate3D sizePML,
                 GridCoordinate3D sizeScatteredZone,
                 FPValue incidentWaveAngle1, /**< teta */
                 FPValue incidentWaveAngle2, /**< phi */
                 FPValue incidentWaveAngle3) /**< psi */
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
  , sizeEps (coordSize + GridCoordinate3D (1, 1, 1))
  , sizeMu (coordSize + GridCoordinate3D (1, 1, 1))
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
  , zeroIncCoordFP (GridCoordinateFP3D (leftBorderTotalField.getX () - 2.5 * sin (incidentWaveAngle1) * cos (incidentWaveAngle2),
                                        leftBorderTotalField.getY () - 2.5 * sin (incidentWaveAngle1) * sin (incidentWaveAngle2),
                                        leftBorderTotalField.getZ () - 2.5 * cos (incidentWaveAngle1)))
  {
    ASSERT (incidentWaveAngle1 == PhysicsConst::Pi / 2);
    ASSERT (incidentWaveAngle2 == 0);
    ASSERT (incidentWaveAngle3 == 0);

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
}; /* YeeGridLayout */

#endif /* YEE_GRID_LAYOUT_H */

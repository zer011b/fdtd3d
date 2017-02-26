#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"

#ifdef CXX11_ENABLED

/**
 * Direction in which to get circut elements
 *
 * FIXME: add base type of uint32_t
 */
enum class LayoutDirection
{
  LEFT, /**< left by Ox */
  RIGHT, /**< right by Ox */
  DOWN, /**< left by Oy */
  UP, /**< right by Oy */
  BACK, /**< left by Oz */
  FRONT /**< right by Oz */
}; /* LayoutDirection */

#else /* CXX11_ENABLED */

/**
 * Direction in which to get circut elements.
 *
 * Note: simple implementation of 'enum class' without C++11 support
 */
class LayoutDirection
{
public:

  /**
   * Direction in which to get circut elements
   */
  enum LayoutDir
  {
    LEFT, /**< left by Ox */
    RIGHT, /**< right by Ox */
    DOWN, /**< left by Oy */
    UP, /**< right by Oy */
    BACK, /**< left by Oz */
    FRONT /**< right by Oz */
  }; /* LayoutDir */

  /**
   * Default constructor
   */
  LayoutDirection (LayoutDir new_dir) /**< direction in which to get circuit element */
    : dir (new_dir)
  {
  } /* LayoutDirection */

  /**
   * Cast operator to integer
   *
   * FIXME: replace with cast operator to uint32_t
   */
  operator int ()
  {
    return dir;
  } /* operator int */

private:

  /** Direction in which to get circut elements */
  LayoutDir dir;
}; /* LayoutDirection */

#endif /* !CXX11_ENABLED */

/**
 * Interface for grid layout which specifies how field components are placed in space
 */
class GridLayout
{
public:

  GridLayout () {}
  virtual ~GridLayout () {}

  /*
   * Get coordinate of circut field component
   */
  virtual GridCoordinate3D getExCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getEyCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getEzCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getHxCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getHyCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;
  virtual GridCoordinate3D getHzCircuitElement (GridCoordinate3D, LayoutDirection) const = 0;

  /*
   * Get size of field component grid
   */
  virtual GridCoordinate3D getEpsSize () const = 0;
  virtual GridCoordinate3D getMuSize () const = 0;
  virtual GridCoordinate3D getExSize () const = 0;
  virtual GridCoordinate3D getEySize () const = 0;
  virtual GridCoordinate3D getEzSize () const = 0;
  virtual GridCoordinate3D getHxSize () const = 0;
  virtual GridCoordinate3D getHySize () const = 0;
  virtual GridCoordinate3D getHzSize () const = 0;

  virtual GridCoordinate3D getSizePML () const = 0;
  virtual GridCoordinate3D getSizeTFSF () const = 0;

  /*
   * Get start coordinate of field component
   */
  virtual GridCoordinate3D getExStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEyStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEzStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHxStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHyStart (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHzStart (GridCoordinate3D) const = 0;

  /*
   * Get end coordinate of field component
   */
  virtual GridCoordinate3D getExEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEyEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getEzEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHxEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHyEnd (GridCoordinate3D) const = 0;
  virtual GridCoordinate3D getHzEnd (GridCoordinate3D) const = 0;

  /*
   * Get minimum coordinate of field component
   */
  virtual GridCoordinate3D getZeroCoord () const = 0;
  virtual GridCoordinate3D getMinEpsCoord () const = 0;
  virtual GridCoordinate3D getMinMuCoord () const = 0;
  virtual GridCoordinate3D getMinExCoord () const = 0;
  virtual GridCoordinate3D getMinEyCoord () const = 0;
  virtual GridCoordinate3D getMinEzCoord () const = 0;
  virtual GridCoordinate3D getMinHxCoord () const = 0;
  virtual GridCoordinate3D getMinHyCoord () const = 0;
  virtual GridCoordinate3D getMinHzCoord () const = 0;

  /*
   * Get minimum real coordinate of field component
   */
  virtual GridCoordinateFP3D getZeroCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinEpsCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinMuCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinExCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinEyCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinEzCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinHxCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinHyCoordFP () const = 0;
  virtual GridCoordinateFP3D getMinHzCoordFP () const = 0;

  /*
   * Get real coordinate of field component by its coordinate
   */
  virtual GridCoordinateFP3D getEpsCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getMuCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getExCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getEyCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getEzCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getHxCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getHyCoordFP (GridCoordinate3D) const = 0;
  virtual GridCoordinateFP3D getHzCoordFP (GridCoordinate3D) const = 0;

  /*
   * Get coordinate of field component by its real coordinate
   */
  virtual GridCoordinate3D getEpsCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getMuCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getExCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getEyCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getEzCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getHxCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getHyCoord (GridCoordinateFP3D) const = 0;
  virtual GridCoordinate3D getHzCoord (GridCoordinateFP3D) const = 0;

  /*
   * PML
   */
  virtual GridCoordinate3D getLeftBorderPML () const = 0;
  virtual GridCoordinate3D getRightBorderPML () const = 0;
  virtual bool isInPML (GridCoordinateFP3D) const = 0;
  virtual bool isExInPML (GridCoordinate3D) const = 0;
  virtual bool isEyInPML (GridCoordinate3D) const = 0;
  virtual bool isEzInPML (GridCoordinate3D) const = 0;
  virtual bool isHxInPML (GridCoordinate3D) const = 0;
  virtual bool isHyInPML (GridCoordinate3D) const = 0;
  virtual bool isHzInPML (GridCoordinate3D) const = 0;

  /*
   * Total field / scattered field
   */
  virtual GridCoordinate3D getLeftBorderTFSF () const = 0;
  virtual GridCoordinate3D getRightBorderTFSF () const = 0;
  virtual GridCoordinateFP3D getZeroIncCoordFP () const = 0;
  virtual bool doNeedTFSFUpdateExBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
  virtual bool doNeedTFSFUpdateEyBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
  virtual bool doNeedTFSFUpdateEzBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
  virtual bool doNeedTFSFUpdateHxBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
  virtual bool doNeedTFSFUpdateHyBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
  virtual bool doNeedTFSFUpdateHzBorder (GridCoordinate3D, LayoutDirection, bool) const = 0;
}; /* GridLayout */

#endif /* GRID_LAYOUT_H */

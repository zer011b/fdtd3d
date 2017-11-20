#include "YeeGridLayout.h"

template <>
const bool YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isParallel = false;
template <>
const bool YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isParallel = false;
template <>
const bool YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::isParallel = false;

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getExCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfExBorderDownX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderUpX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderBackX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderFrontX (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getExCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfExBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderDownY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderUpY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderBackY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderFrontY (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getExCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfExBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderDownY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderDownZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderUpY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderUpZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderBackY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderBackZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfExBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderFrontY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfExBorderFrontZ (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getEyCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEyBorderLeftX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderRightX (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfEyBorderBackX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderFrontX (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getEyCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEyBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderLeftY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderRightY (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfEyBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderBackY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderFrontY (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getEyCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEyBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderLeftY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderLeftZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderRightY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderRightZ (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfEyBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderBackY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderBackZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEyBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderFrontY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEyBorderFrontZ (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getEzCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEzBorderLeftX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderRightX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderDownX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderUpX (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getEzCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEzBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderLeftY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderRightY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderDownY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderUpY (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getEzCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfEzBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderLeftY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderLeftZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderRightY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderRightZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderDownY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderDownZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfEzBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderUpY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfEzBorderUpZ (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getHxCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHxBorderDownX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderUpX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderBackX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderFrontX (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getHxCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHxBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderDownY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderUpY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderBackY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderFrontY (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getHxCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHxBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderDownY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderDownZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderUpY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderUpZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderBackY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderBackZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHxBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderFrontY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHxBorderFrontZ (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getHyCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHyBorderLeftX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderRightX (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHyBorderBackX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderFrontX (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getHyCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHyBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderLeftY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderRightY (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHyBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderBackY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderFrontY (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getHyCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHyBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderLeftY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderLeftZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderRightY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderRightZ (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfHyBorderBackX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderBackY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderBackZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHyBorderFrontX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderFrontY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHyBorderFrontZ (coordFP, leftBorderFP, rightBorderFP));
}

template <>
bool
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand2 (getHzCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand2 (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand2 (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHzBorderLeftX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderRightX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderDownX (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderUpX (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = expand (getHzCoordFP (coord));

  GridCoordinateFP3D leftBorderFP = expand (zeroCoordFP + convertCoord (leftBorderTotalField));
  GridCoordinateFP3D rightBorderFP = expand (zeroCoordFP + convertCoord (rightBorderTotalField));

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHzBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderLeftY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderRightY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderDownY (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderUpY (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  GridCoordinateFP3D coordFP = getHzCoordFP (coord);

  GridCoordinateFP3D leftBorderFP = zeroCoordFP + convertCoord (leftBorderTotalField);
  GridCoordinateFP3D rightBorderFP = zeroCoordFP + convertCoord (rightBorderTotalField);

  return doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfHzBorderLeftX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderLeftY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderLeftZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderRightX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderRightY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderRightZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderDownX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderDownY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderDownZ (coordFP, leftBorderFP, rightBorderFP),
                                 YeeGridLayoutHelper::tfsfHzBorderUpX (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderUpY (coordFP, leftBorderFP, rightBorderFP)
                                 && YeeGridLayoutHelper::tfsfHzBorderUpZ (coordFP, leftBorderFP, rightBorderFP),
                                 false,
                                 false);
}

template <>
bool YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  GridCoordinateFP1D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
  GridCoordinateFP1D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

  /*
   * TODO: remove floating point equality comparison
   */
  bool isInXPML = coordLeftBorderPMLFP.getX () != coordRightBorderPMLFP.getX ()
                  && (realCoordFP.getX () < coordLeftBorderPMLFP.getX ()
                      || realCoordFP.getX () >= coordRightBorderPMLFP.getX ());

  return isInXPML;
}

template <>
bool YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  GridCoordinateFP2D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
  GridCoordinateFP2D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

  /*
   * TODO: remove floating point equality comparison
   */
  bool isInXPML = coordLeftBorderPMLFP.getX () != coordRightBorderPMLFP.getX ()
                  && (realCoordFP.getX () < coordLeftBorderPMLFP.getX ()
                      || realCoordFP.getX () >= coordRightBorderPMLFP.getX ());
  bool isInYPML = coordLeftBorderPMLFP.getY () != coordRightBorderPMLFP.getY ()
                  && (realCoordFP.getY () < coordLeftBorderPMLFP.getY ()
                      || realCoordFP.getY () >= coordRightBorderPMLFP.getY ());

  return isInXPML || isInYPML;
}

template <>
bool YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP3D realCoordFP) const
{
  GridCoordinateFP3D coordLeftBorderPMLFP = convertCoord (leftBorderPML) + zeroCoordFP;
  GridCoordinateFP3D coordRightBorderPMLFP = convertCoord (rightBorderPML) + zeroCoordFP;

  /*
   * TODO: remove floating point equality comparison
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


#define DEFAULT_VALS_LIST \
  zeroCoord (0, 0, 0) \
  , minEpsCoord (0, 0, 0) \
  , minMuCoord (0, 0, 0) \
  , minExCoord (0, 0, 0) \
  , minEyCoord (0, 0, 0) \
  , minEzCoord (0, 0, 0) \
  , minHxCoord (0, 0, 0) \
  , minHyCoord (0, 0, 0) \
  , minHzCoord (0, 0, 0) \
  , minEpsCoordFP (0.5, 0.5, 0.5) \
  , minMuCoordFP (0.5, 0.5, 0.5) \
  , zeroCoordFP (0.0, 0.0, 0.0) \
  , minExCoordFP (1.0, 0.5, 0.5) \
  , minEyCoordFP (0.5, 1.0, 0.5) \
  , minEzCoordFP (0.5, 0.5, 1.0) \
  , minHxCoordFP (0.5, 1.0, 1.0) \
  , minHyCoordFP (1.0, 0.5, 1.0) \
  , minHzCoordFP (1.0, 1.0, 0.5) \
  , size (coordSize) \
  , sizeEps (coordSize * (doubleMaterialPrecision ? 2 : 1)) \
  , sizeMu (coordSize * (doubleMaterialPrecision ? 2 : 1)) \
  , sizeEx (coordSize) \
  , sizeEy (coordSize) \
  , sizeEz (coordSize) \
  , sizeHx (coordSize) \
  , sizeHy (coordSize) \
  , sizeHz (coordSize) \
  , leftBorderPML (sizePML) \
  , rightBorderPML (coordSize - sizePML) \
  , leftBorderTotalField (sizeScatteredZone) \
  , rightBorderTotalField (coordSize - sizeScatteredZone) \
  , incidentWaveAngle1 (incWaveAngle1) \
  , incidentWaveAngle2 (incWaveAngle2) \
  , incidentWaveAngle3 (incWaveAngle3) \
  , isDoubleMaterialPrecision (doubleMaterialPrecision)

template <>
YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout (GridCoordinate1D coordSize,
                                                                                GridCoordinate1D sizePML,
                                                                                GridCoordinate1D sizeScatteredZone,
                                                                                FPValue incWaveAngle1, /**< teta */
                                                                                FPValue incWaveAngle2, /**< phi */
                                                                                FPValue incWaveAngle3, /**< psi */
                                                                                bool doubleMaterialPrecision)
: DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.getX () - 2.5 * sin (incWaveAngle1) * cos (incWaveAngle2))
{
  ASSERT (size.getX () > 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout (GridCoordinate2D coordSize,
                                                                                GridCoordinate2D sizePML,
                                                                                GridCoordinate2D sizeScatteredZone,
                                                                                FPValue incWaveAngle1, /**< teta */
                                                                                FPValue incWaveAngle2, /**< phi */
                                                                                FPValue incWaveAngle3, /**< psi */
                                                                                bool doubleMaterialPrecision)
: DEFAULT_VALS_LIST
// TODO: check this, only two angles should be used!
, zeroIncCoordFP (leftBorderTotalField.getX () - 2.5 * sin (incWaveAngle1) * cos (incWaveAngle2),
                  leftBorderTotalField.getY () - 2.5 * sin (incWaveAngle1) * sin (incWaveAngle2))
{
  ASSERT (size.getX () > 0);
  ASSERT (size.getY () > 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED>::YeeGridLayout (GridCoordinate3D coordSize,
                                                                                GridCoordinate3D sizePML,
                                                                                GridCoordinate3D sizeScatteredZone,
                                                                                FPValue incWaveAngle1, /**< teta */
                                                                                FPValue incWaveAngle2, /**< phi */
                                                                                FPValue incWaveAngle3, /**< psi */
                                                                                bool doubleMaterialPrecision)
: DEFAULT_VALS_LIST
// TODO: check this, only two angles should be used!
, zeroIncCoordFP (leftBorderTotalField.getX () - 2.5 * sin (incWaveAngle1) * cos (incWaveAngle2),
                  leftBorderTotalField.getY () - 2.5 * sin (incWaveAngle1) * sin (incWaveAngle2),
                  leftBorderTotalField.getZ () - 2.5 * cos (incWaveAngle1))
{
  ASSERT (size.getX () > 0);
  ASSERT (size.getY () > 0);
  ASSERT (size.getZ () > 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
}

#undef DEFAULT_VALS_LIST

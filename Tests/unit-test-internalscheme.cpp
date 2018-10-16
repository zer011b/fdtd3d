/*
 * Unit test for basic operations with Grid
 */

#include <iostream>

#include "InternalScheme.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
YeeGridLayout<Type, TCoord, layout_type> *
getLayout (CoordinateType ct1, CoordinateType ct2, CoordinateType ct3,
           FPValue angle1, FPValue angle2, FPValue angle3)
{
  TCoord<grid_coord, true> overallSize (20, 20, 20, ct1, ct2, ct3);
  TCoord<grid_coord, true> pmlSize (5, 5, 5, ct1, ct2, ct3);
  TCoord<grid_coord, true> tfsfSizeLeft (7, 7, 7, ct1, ct2, ct3);
  TCoord<grid_coord, true> tfsfSizeRight (13, 13, 13, ct1, ct2, ct3);
  bool useDoubleMaterialPrecision = false;

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout = new YeeGridLayout<Type, TCoord, layout_type> (
              overallSize,
              pmlSize,
              tfsfSizeLeft,
              tfsfSizeRight,
              angle1 * PhysicsConst::Pi / 180.0,
              angle2 * PhysicsConst::Pi / 180.0,
              angle3 * PhysicsConst::Pi / 180.0,
              useDoubleMaterialPrecision);

  return yeeLayout;
}

int main (int argc, char** argv)
{
  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED> *yeeLayout =
    getLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>
    (CoordinateType::Z, CoordinateType::NONE, CoordinateType::NONE, 0, 0, 90);

  InternalScheme1D_ExHy_Grid<E_CENTERED> intScheme;
  intScheme.init (yeeLayout, false);
  intScheme.performFieldSteps<static_cast<uint8_t> (GridType::EX)> (1, GridCoordinate1D (0, CoordinateType::Z), GridCoordinate1D (20, CoordinateType::Z));
  intScheme.performPlaneWaveESteps (1);

  return 0;
} /* main */

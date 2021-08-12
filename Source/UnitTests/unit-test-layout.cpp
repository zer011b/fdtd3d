/*
 * Copyright (C) 2018 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * Unit test for basic operations with GridCoordinate
 */

#include <iostream>

#include "Assert.h"
#include "YeeGridLayout.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

#define SIZEMULT 5
#define SIZEX 3
#define SIZEY 5
#define SIZEZ 4
#define MULT 2

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void testFuncInternal (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sx, grid_coord sy, grid_coord sz)
{
  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  switch (Type)
  {
    case (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)):
    {
      ct1 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)):
    {
      ct1 = CoordinateType::Y;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)):
    {
      ct1 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)):
    {
      ct1 = CoordinateType::X;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)):
    {
      ct1 = CoordinateType::Y;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)):
    {
      ct1 = CoordinateType::X;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)):
    {
      ct1 = CoordinateType::Y;
      ct2 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)):
    {
      ct1 = CoordinateType::X;
      ct2 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)):
    {
      ct1 = CoordinateType::X;
      ct2 = CoordinateType::Y;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)):
    {
      ct1 = CoordinateType::Y;
      ct2 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)):
    {
      ct1 = CoordinateType::X;
      ct2 = CoordinateType::Z;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)):
    {
      ct1 = CoordinateType::X;
      ct2 = CoordinateType::Y;
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim3)):
    {
      ct1 = CoordinateType::X;
      ct2 = CoordinateType::Y;
      ct3 = CoordinateType::Z;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizez = sz * SIZEMULT;

  grid_coord sizex_h = sizex / 2;
  grid_coord sizey_h = sizey / 2;
  grid_coord sizez_h = sizez / 2;

  grid_coord sizex_tfsf = sx * mult;
  grid_coord sizey_tfsf = sy * mult;
  grid_coord sizez_tfsf = sz * mult;

  TCoord<grid_coord, true> size = TCoord<grid_coord, true>::initAxesCoordinate (sizex, sizey, sizez, ct1, ct2, ct3);
  TCoord<grid_coord, true> sizePML = TCoord<grid_coord, true>::initAxesCoordinate (sx, sy, sz, ct1, ct2, ct3);
  TCoord<grid_coord, true> sizeTFSFLeft = TCoord<grid_coord, true>::initAxesCoordinate (sizex_tfsf, sizey_tfsf, sizez_tfsf, ct1, ct2, ct3);
  TCoord<grid_coord, true> sizeTFSFRight = TCoord<grid_coord, true>::initAxesCoordinate (sizex_tfsf * 2, sizey_tfsf * 2, sizez_tfsf * 2, ct1, ct2, ct3);

  YeeGridLayout<Type, TCoord, layout_type> layout (size,
                                                   sizePML,
                                                   sizeTFSFLeft,
                                                   sizeTFSFRight,
                                                   incAngle1,
                                                   incAngle2,
                                                   incAngle3,
                                                   doubleMaterialPrecision);
  ALWAYS_ASSERT (layout.getEpsSize () == size * (doubleMaterialPrecision ? 2 : 1));
  ALWAYS_ASSERT (layout.getMuSize () == size * (doubleMaterialPrecision ? 2 : 1));

  ALWAYS_ASSERT (layout.getSigmaXSize () == layout.getEpsSize ());
  ALWAYS_ASSERT (layout.getSigmaYSize () == layout.getEpsSize ());
  ALWAYS_ASSERT (layout.getSigmaZSize () == layout.getEpsSize ());

  ALWAYS_ASSERT (layout.getExSize () == size);
  ALWAYS_ASSERT (layout.getEySize () == size);
  ALWAYS_ASSERT (layout.getEzSize () == size);
  ALWAYS_ASSERT (layout.getHxSize () == size);
  ALWAYS_ASSERT (layout.getHySize () == size);
  ALWAYS_ASSERT (layout.getHzSize () == size);

  ALWAYS_ASSERT (layout.getSizePML () == sizePML);
  ALWAYS_ASSERT (layout.getLeftBorderTFSF () == sizeTFSFLeft);
  ALWAYS_ASSERT (layout.getRightBorderTFSF () == size - sizeTFSFRight);

  ALWAYS_ASSERT ((layout.getExStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getEyStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getEzStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHxStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHyStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHzStartDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));

  ALWAYS_ASSERT ((layout.getExEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getEyEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getEzEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHxEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHyEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getHzEndDiff () == TCoord<grid_coord, true>::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3)));

  ALWAYS_ASSERT ((layout.getZeroCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getMinEpsCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getMinMuCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)));

  ALWAYS_ASSERT (layout.getLeftBorderPML () == layout.getSizePML ());
  ALWAYS_ASSERT (layout.getRightBorderPML () + layout.getSizePML () == layout.getSize ());

  ALWAYS_ASSERT (layout.getIncidentWaveAngle1 () == incAngle1);
  ALWAYS_ASSERT (layout.getIncidentWaveAngle2 () == incAngle2);
  ALWAYS_ASSERT (layout.getIncidentWaveAngle3 () == incAngle3);

  ALWAYS_ASSERT (layout.getIsDoubleMaterialPrecision () == doubleMaterialPrecision);

  TCoord<grid_coord, true> coordMaterial = size / grid_coord (2);
  ALWAYS_ASSERT ((layout.getEpsCoordFP (coordMaterial) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getMuCoordFP (coordMaterial) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)));
  ALWAYS_ASSERT (layout.getEpsCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)) == coordMaterial);
  ALWAYS_ASSERT (layout.getMuCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)) == coordMaterial);
  ALWAYS_ASSERT ((layout.getMinEpsCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)));
  ALWAYS_ASSERT ((layout.getMinMuCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)));

  TCoord<grid_coord, true> coordEx = size / grid_coord (2);
  TCoord<grid_coord, true> coordHzD;
  TCoord<grid_coord, true> coordHzU;
  TCoord<grid_coord, true> coordHyB;
  TCoord<grid_coord, true> coordHyF;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinExCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 0.5, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getExCoordFP (coordEx) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getExCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)) == coordEx);
    coordHzD = coordEx - TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordHzU = coordEx;
    coordHyB = coordEx - TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
    coordHyF = coordEx;
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinExCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 1, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getExCoordFP (coordEx) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getExCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 1, ct1, ct2, ct3)) == coordEx);
    coordHzD = coordEx;
    coordHzU = coordEx + TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordHyB = coordEx;
    coordHyF = coordEx + TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordEx + layout.getExCircuitElementDiff (LayoutDirection::DOWN) == coordHzD);
  ALWAYS_ASSERT (coordEx + layout.getExCircuitElementDiff (LayoutDirection::UP) == coordHzU);
  ALWAYS_ASSERT (coordEx + layout.getExCircuitElementDiff (LayoutDirection::BACK) == coordHyB);
  ALWAYS_ASSERT (coordEx + layout.getExCircuitElementDiff (LayoutDirection::FRONT) == coordHyF);

  TCoord<grid_coord, true> coordEy = size / grid_coord (2);
  TCoord<grid_coord, true> coordHzL;
  TCoord<grid_coord, true> coordHzR;
  TCoord<grid_coord, true> coordHxB;
  TCoord<grid_coord, true> coordHxF;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinEyCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 1, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getEyCoordFP (coordEy) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getEyCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)) == coordEy);
    coordHzL = coordEy - TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordHzR = coordEy;
    coordHxB = coordEy - TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
    coordHxF = coordEy;
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinEyCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 0.5, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getEyCoordFP (coordEy) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getEyCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)) == coordEy);
    coordHzL = coordEy;
    coordHzR = coordEy + TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordHxB = coordEy;
    coordHxF = coordEy + TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordEy + layout.getEyCircuitElementDiff (LayoutDirection::LEFT) == coordHzL);
  ALWAYS_ASSERT (coordEy + layout.getEyCircuitElementDiff (LayoutDirection::RIGHT) == coordHzR);
  ALWAYS_ASSERT (coordEy + layout.getEyCircuitElementDiff (LayoutDirection::BACK) == coordHxB);
  ALWAYS_ASSERT (coordEy + layout.getEyCircuitElementDiff (LayoutDirection::FRONT) == coordHxF);

  TCoord<grid_coord, true> coordEz = size / grid_coord (2);
  TCoord<grid_coord, true> coordHyL;
  TCoord<grid_coord, true> coordHyR;
  TCoord<grid_coord, true> coordHxD;
  TCoord<grid_coord, true> coordHxU;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinEzCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getEzCoordFP (coordEz) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getEzCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)) == coordEz);
    coordHyL = coordEz - TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordHyR = coordEz;
    coordHxD = coordEz - TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordHxU = coordEz;
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinEzCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 1, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getEzCoordFP (coordEz) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getEzCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)) == coordEz);
    coordHyL = coordEz;
    coordHyR = coordEz + TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordHxD = coordEz;
    coordHxU = coordEz + TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordEz + layout.getEzCircuitElementDiff (LayoutDirection::LEFT) == coordHyL);
  ALWAYS_ASSERT (coordEz + layout.getEzCircuitElementDiff (LayoutDirection::RIGHT) == coordHyR);
  ALWAYS_ASSERT (coordEz + layout.getEzCircuitElementDiff (LayoutDirection::DOWN) == coordHxD);
  ALWAYS_ASSERT (coordEz + layout.getEzCircuitElementDiff (LayoutDirection::UP) == coordHxU);

  TCoord<grid_coord, true> coordHx = size / grid_coord (2);
  TCoord<grid_coord, true> coordEzD;
  TCoord<grid_coord, true> coordEzU;
  TCoord<grid_coord, true> coordEyB;
  TCoord<grid_coord, true> coordEyF;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHxCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 1, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHxCoordFP (coordHx) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHxCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 1, ct1, ct2, ct3)) == coordHx);
    coordEzD = coordHx;
    coordEzU = coordHx + TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordEyB = coordHx;
    coordEyF = coordHx + TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHxCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 0.5, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHxCoordFP (coordHx) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHxCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 0.5, ct1, ct2, ct3)) == coordHx);
    coordEzD = coordHx - TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordEzU = coordHx;
    coordEyB = coordHx - TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
    coordEyF = coordHx;
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordHx + layout.getHxCircuitElementDiff (LayoutDirection::DOWN) == coordEzD);
  ALWAYS_ASSERT (coordHx + layout.getHxCircuitElementDiff (LayoutDirection::UP) == coordEzU);
  ALWAYS_ASSERT (coordHx + layout.getHxCircuitElementDiff (LayoutDirection::BACK) == coordEyB);
  ALWAYS_ASSERT (coordHx + layout.getHxCircuitElementDiff (LayoutDirection::FRONT) == coordEyF);


  TCoord<grid_coord, true> coordHy = size / grid_coord (2);
  TCoord<grid_coord, true> coordEzL;
  TCoord<grid_coord, true> coordEzR;
  TCoord<grid_coord, true> coordExB;
  TCoord<grid_coord, true> coordExF;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHyCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 0.5, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHyCoordFP (coordHy) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHyCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)) == coordHy);
    coordEzL = coordHy;
    coordEzR = coordHy + TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordExB = coordHy;
    coordExF = coordHy + TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHyCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 1, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHyCoordFP (coordHy) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHyCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)) == coordHy);
    coordEzL = coordHy - TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordEzR = coordHy;
    coordExB = coordHy - TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 1, ct1, ct2, ct3);
    coordExF = coordHy;
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordHy + layout.getHyCircuitElementDiff (LayoutDirection::LEFT) == coordEzL);
  ALWAYS_ASSERT (coordHy + layout.getHyCircuitElementDiff (LayoutDirection::RIGHT) == coordEzR);
  ALWAYS_ASSERT (coordHy + layout.getHyCircuitElementDiff (LayoutDirection::BACK) == coordExB);
  ALWAYS_ASSERT (coordHy + layout.getHyCircuitElementDiff (LayoutDirection::FRONT) == coordExF);

  TCoord<grid_coord, true> coordHz = size / grid_coord (2);
  TCoord<grid_coord, true> coordEyL;
  TCoord<grid_coord, true> coordEyR;
  TCoord<grid_coord, true> coordExD;
  TCoord<grid_coord, true> coordExU;
  if (layout_type == E_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHzCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (1, 1, 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHzCoordFP (coordHz) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHzCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 1, sizey_h + 1, sizez_h + 0.5, ct1, ct2, ct3)) == coordHz);
    coordEyL = coordHz;
    coordEyR = coordHz + TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordExD = coordHz;
    coordExU = coordHz + TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
  }
  else if (layout_type == H_CENTERED)
  {
    ALWAYS_ASSERT ((layout.getMinHzCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.5, 0.5, 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT ((layout.getHzCoordFP (coordHz) == TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)));
    ALWAYS_ASSERT (layout.getHzCoord (TCoord<FPValue, true>::initAxesCoordinate (sizex_h + 0.5, sizey_h + 0.5, sizez_h + 1, ct1, ct2, ct3)) == coordHz);
    coordEyL = coordHz - TCoord<grid_coord, true>::initAxesCoordinate (1, 0, 0, ct1, ct2, ct3);
    coordEyR = coordHz;
    coordExD = coordHz - TCoord<grid_coord, true>::initAxesCoordinate (0, 1, 0, ct1, ct2, ct3);
    coordExU = coordHz;
  }
  else
  {
    UNREACHABLE;
  }
  ALWAYS_ASSERT (coordHz + layout.getHzCircuitElementDiff (LayoutDirection::LEFT) == coordEyL);
  ALWAYS_ASSERT (coordHz + layout.getHzCircuitElementDiff (LayoutDirection::RIGHT) == coordEyR);
  ALWAYS_ASSERT (coordHz + layout.getHzCircuitElementDiff (LayoutDirection::DOWN) == coordExD);
  ALWAYS_ASSERT (coordHz + layout.getHzCircuitElementDiff (LayoutDirection::UP) == coordExU);

  switch (Type)
  {
    case (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         0.0,
                                                                                         sizez_tfsf - 2.5, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         sizey_tfsf - 2.5,
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         0.0,
                                                                                         sizez_tfsf - 2.5, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5,
                                                                                         0.0,
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         sizey_tfsf - 2.5,
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5,
                                                                                         0.0,
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         sizey_tfsf - 2.5 * sin (incAngle1),
                                                                                         sizez_tfsf - 2.5 * cos (incAngle1), ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- cos (incAngle1)));
      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle1)));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5 * sin (incAngle1),
                                                                                         0.0,
                                                                                         sizez_tfsf - 2.5 * cos (incAngle1), ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- cos (incAngle1)));
      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle1)));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5 * cos (incAngle2),
                                                                                         sizey_tfsf - 2.5 * sin (incAngle2),
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle2)));
      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) ( - cos (incAngle2)));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (0.0,
                                                                                         sizey_tfsf - 2.5 * sin (incAngle1),
                                                                                         sizez_tfsf - 2.5 * cos (incAngle1), ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (cos (incAngle1)));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- sin (incAngle1)));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5 * sin (incAngle1),
                                                                                         0.0,
                                                                                         sizez_tfsf - 2.5 * cos (incAngle1), ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == - FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (cos (incAngle1)));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- sin (incAngle1)));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5 * cos (incAngle2),
                                                                                         sizey_tfsf - 2.5 * sin (incAngle2),
                                                                                         0.0, ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle2)));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- cos (incAngle2)));
      break;
    }
    case (static_cast<SchemeType_t> (SchemeType::Dim3)):
    {
      ALWAYS_ASSERT ((layout.getZeroIncCoordFP () == TCoord<FPValue, true>::initAxesCoordinate (sizex_tfsf - 2.5 * sin (incAngle1) * cos (incAngle2),
                                                                                         sizey_tfsf - 2.5 * sin (incAngle1) * sin (incAngle2),
                                                                                         sizez_tfsf - 2.5 * cos (incAngle1), ct1, ct2, ct3)));

      ALWAYS_ASSERT (layout.getExFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (cos (incAngle3) * sin (incAngle2) - sin (incAngle3) * cos (incAngle1) * cos (incAngle2)));
      ALWAYS_ASSERT (layout.getEyFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) ( - cos (incAngle3) * cos (incAngle2) - sin (incAngle3) * cos (incAngle1) * sin (incAngle2)));
      ALWAYS_ASSERT (layout.getEzFromIncidentE (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle3) * sin (incAngle1)));
      ALWAYS_ASSERT (layout.getHxFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (sin (incAngle3) * sin (incAngle2) + cos (incAngle3) * cos (incAngle1) * cos (incAngle2)));
      ALWAYS_ASSERT (layout.getHyFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- sin (incAngle3) * cos (incAngle2) + cos (incAngle3) * cos (incAngle1) * sin (incAngle2)));
      ALWAYS_ASSERT (layout.getHzFromIncidentH (FIELDVALUE (17.0, 53.0)) == FIELDVALUE (17.0, 53.0) * (FPValue) (- cos (incAngle3) * sin (incAngle1)));
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  /*
   * TODO: add next
   *       getApproximateMaterial
   *       getApproximateMetaMaterial
   *       getMetaMaterial
   *       getMaterial
   *       initMaterialCoordinates
   */
}

#define INITIALIZATION \
  bool isExInPML = layout.isExInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  bool isEyInPML = layout.isEyInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  bool isEzInPML = layout.isEzInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  bool isHxInPML = layout.isHxInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  bool isHyInPML = layout.isHyInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  bool isHzInPML = layout.isHzInPML (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  \
  bool doNeedUpdateExL = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateExR = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateExD = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateExU = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateExB = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateExF = layout.doNeedTFSFUpdateExBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  bool doNeedUpdateEyL = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateEyR = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateEyD = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateEyU = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateEyB = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateEyF = layout.doNeedTFSFUpdateEyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  bool doNeedUpdateEzL = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateEzR = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateEzD = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateEzU = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateEzB = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateEzF = layout.doNeedTFSFUpdateEzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  bool doNeedUpdateHxL = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateHxR = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateHxD = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateHxU = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateHxB = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateHxF = layout.doNeedTFSFUpdateHxBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  bool doNeedUpdateHyL = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateHyR = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateHyD = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateHyU = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateHyB = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateHyF = layout.doNeedTFSFUpdateHyBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  bool doNeedUpdateHzL = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::LEFT); \
  bool doNeedUpdateHzR = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::RIGHT); \
  bool doNeedUpdateHzD = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::DOWN); \
  bool doNeedUpdateHzU = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::UP); \
  bool doNeedUpdateHzB = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::BACK); \
  bool doNeedUpdateHzF = layout.doNeedTFSFUpdateHzBorder (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3), LayoutDirection::FRONT); \
  \
  TCoord<FPValue, true> coordExFP = layout.getExCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  TCoord<FPValue, true> coordEyFP = layout.getEyCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  TCoord<FPValue, true> coordEzFP = layout.getEzCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  TCoord<FPValue, true> coordHxFP = layout.getHxCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  TCoord<FPValue, true> coordHyFP = layout.getHyCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \
  TCoord<FPValue, true> coordHzFP = layout.getHzCoordFP (TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3)); \

#if defined (MODE_EX_HY)

template<LayoutType layout_type>
void testFuncDim1_ExHy (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("ExHy\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Z;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  grid_coord sizez_h = sizez / 2;

  GridCoordinate1D size (sizez, ct1);
  GridCoordinate1D sizePML (sz, ct1);
  GridCoordinate1D sizeTFSFLeft (sizez_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizez_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sz || coordExFP.get1 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordHyFP.get1 () < sz || coordHyFP.get1 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR && !doNeedUpdateExD && !doNeedUpdateExU);
    ALWAYS_ASSERT (!doNeedUpdateHyL && !doNeedUpdateHyR && !doNeedUpdateHyD && !doNeedUpdateHyU);

    if (layout_type == E_CENTERED)
    {
      if (coordExFP.get1 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordHyFP.get1 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordExFP.get1 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordHyFP.get1 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EX_HY */

#if defined (MODE_EX_HZ)

template<LayoutType layout_type>
void testFuncDim1_ExHz (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sy)
{
#if PRINT_MESSAGE
  printf ("ExHz\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  GridCoordinate1D size (sizey, ct1);
  GridCoordinate1D sizePML (sy, ct1);
  GridCoordinate1D sizeTFSFLeft (sizey_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizey_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sy || coordExFP.get1 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordHzFP.get1 () < sy || coordHzFP.get1 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR && !doNeedUpdateExB && !doNeedUpdateExF);
    ALWAYS_ASSERT (!doNeedUpdateHzL && !doNeedUpdateHzR && !doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      if (coordExFP.get1 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordHzFP.get1 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordExFP.get1 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordHzFP.get1 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EX_HZ */

#if defined (MODE_EY_HX)

template<LayoutType layout_type>
void testFuncDim1_EyHx (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("EyHx\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Z;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate1D size (sizez, ct1);
  GridCoordinate1D sizePML (sz, ct1);
  GridCoordinate1D sizeTFSFLeft (sizez_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizez_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEyFP.get1 () < sz || coordEyFP.get1 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordHxFP.get1 () < sz || coordHxFP.get1 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEyL && !doNeedUpdateEyR && !doNeedUpdateEyD && !doNeedUpdateEyU);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR && !doNeedUpdateHxD && !doNeedUpdateHxU);

    if (layout_type == E_CENTERED)
    {
      if (coordEyFP.get1 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordHxFP.get1 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEyFP.get1 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordHxFP.get1 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EY_HX */

#if defined (MODE_EY_HZ)

template<LayoutType layout_type>
void testFuncDim1_EyHz (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sx)
{
#if PRINT_MESSAGE
  printf ("EyHz\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  GridCoordinate1D size (sizex, ct1);
  GridCoordinate1D sizePML (sx, ct1);
  GridCoordinate1D sizeTFSFLeft (sizex_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizex_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEyFP.get1 () < sx || coordEyFP.get1 () >= sizex - sx)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordHzFP.get1 () < sx || coordHzFP.get1 () >= sizex - sx)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEyD && !doNeedUpdateEyU && !doNeedUpdateEyB && !doNeedUpdateEyF);
    ALWAYS_ASSERT (!doNeedUpdateHzD && !doNeedUpdateHzU && !doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      if (coordEyFP.get1 () == sizex_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordHzFP.get1 () == sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEyFP.get1 () == sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordHzFP.get1 () == sizex_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EY_HZ */

#if defined (MODE_EZ_HX)

template<LayoutType layout_type>
void testFuncDim1_EzHx (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sy)
{
#if PRINT_MESSAGE
  printf ("EzHx\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  GridCoordinate1D size (sizey, ct1);
  GridCoordinate1D sizePML (sy, ct1);
  GridCoordinate1D sizeTFSFLeft (sizey_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizey_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEzFP.get1 () < sy || coordEzFP.get1 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHxFP.get1 () < sy || coordHxFP.get1 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEzL && !doNeedUpdateEzR && !doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR && !doNeedUpdateHxB && !doNeedUpdateHxF);

    if (layout_type == E_CENTERED)
    {
      if (coordEzFP.get1 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEzFP.get1 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EZ_HX */

#if defined (MODE_EZ_HY)

template<LayoutType layout_type>
void testFuncDim1_EzHy (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                        grid_coord mult, grid_coord sx)
{
#if PRINT_MESSAGE
  printf ("EzHy\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  GridCoordinate1D size (sizex, ct1);
  GridCoordinate1D sizePML (sx, ct1);
  GridCoordinate1D sizeTFSFLeft (sizex_tfsf, ct1);
  GridCoordinate1D sizeTFSFRight (sizex_tfsf, ct1);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, layout_type> layout (size,
                                                                                      sizePML,
                                                                                      sizeTFSFLeft,
                                                                                      sizeTFSFRight,
                                                                                      incAngle1,
                                                                                      incAngle2,
                                                                                      incAngle3,
                                                                                      doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate1DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEzFP.get1 () < sx || coordEzFP.get1 () >= sizex - sx)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHyFP.get1 () < sx || coordHyFP.get1 () >= sizex - sx)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEzD && !doNeedUpdateEzU && !doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHyD && !doNeedUpdateHyU && !doNeedUpdateHyB && !doNeedUpdateHyF);

    if (layout_type == E_CENTERED)
    {
      if (coordEzFP.get1 () == sizex_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordHyFP.get1 () == sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEzFP.get1 () == sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordHyFP.get1 () == sizex_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_EZ_HY */

#if defined (MODE_TEX)

template<LayoutType layout_type>
void testFuncDim2_TEx (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sy, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("TEx\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate2D size (sizey, sizez, ct1, ct2);
  GridCoordinate2D sizePML (sy, sz, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizey_tfsf, sizez_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizey_tfsf, sizez_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEyFP.get1 () < sy || coordEyFP.get1 () >= sizey - sy
        || coordEyFP.get2 () < sz || coordEyFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordEzFP.get1 () < sy || coordEzFP.get1 () >= sizey - sy
        || coordEzFP.get2 () < sz || coordEzFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHxFP.get1 () < sy || coordHxFP.get1 () >= sizey - sy
        || coordHxFP.get2 () < sz || coordHxFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEyL && !doNeedUpdateEyR && !doNeedUpdateEyD && !doNeedUpdateEyU);
    ALWAYS_ASSERT (!doNeedUpdateEzL && !doNeedUpdateEzR && !doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR);

    if (layout_type == E_CENTERED)
    {
      if (coordEyFP.get1 () > sizey_tfsf - 0.1 && coordEyFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizey_tfsf - 0.1 && coordEyFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordEzFP.get1 () == sizey_tfsf - 0.5
          && coordEzFP.get2 () > sizez_tfsf - 0.1 && coordEzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () == sizey - sizey_tfsf + 0.5
          && coordEzFP.get2 () > sizez_tfsf - 0.1 && coordEzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () == sizey_tfsf
          && coordHxFP.get2 () > sizez_tfsf - 0.1 && coordHxFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () == sizey - sizey_tfsf
          && coordHxFP.get2 () > sizez_tfsf - 0.1 && coordHxFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHxFP.get1 () > sizey_tfsf - 0.1 && coordHxFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordHxFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizey_tfsf - 0.1 && coordHxFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordHxFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEyFP.get1 () > sizey_tfsf + 0.4 && coordEyFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizey_tfsf + 0.4 && coordEyFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordEzFP.get1 () == sizey_tfsf
          && coordEzFP.get2 () > sizez_tfsf + 0.4 && coordEzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () == sizey - sizey_tfsf
          && coordEzFP.get2 () > sizez_tfsf + 0.4 && coordEzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () == sizey_tfsf - 0.5
          && coordHxFP.get2 () > sizez_tfsf + 0.4 && coordHxFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () == sizey - sizey_tfsf + 0.5
          && coordHxFP.get2 () > sizez_tfsf + 0.4 && coordHxFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHxFP.get1 () > sizey_tfsf + 0.4 && coordHxFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordHxFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizey_tfsf + 0.4 && coordHxFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordHxFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TEX */

#if defined (MODE_TEY)

template<LayoutType layout_type>
void testFuncDim2_TEy (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sx, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("TEy\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate2D size (sizex, sizez, ct1, ct2);
  GridCoordinate2D sizePML (sx, sz, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizex_tfsf, sizez_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizex_tfsf, sizez_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sx || coordExFP.get1 () >= sizex - sx
        || coordExFP.get2 () < sz || coordExFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordEzFP.get1 () < sx || coordEzFP.get1 () >= sizex - sx
        || coordEzFP.get2 () < sz || coordEzFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHyFP.get1 () < sx || coordHyFP.get1 () >= sizex - sx
        || coordHyFP.get2 () < sz || coordHyFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR && !doNeedUpdateExD && !doNeedUpdateExU);
    ALWAYS_ASSERT (!doNeedUpdateEzD && !doNeedUpdateEzU && !doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHyD && !doNeedUpdateHyU);

    if (layout_type == E_CENTERED)
    {
      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordEzFP.get1 () == sizex_tfsf - 0.5
          && coordEzFP.get2 () > sizez_tfsf - 0.1 && coordEzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEzFP.get2 () > sizez_tfsf - 0.1 && coordEzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordHyFP.get1 () == sizex_tfsf
          && coordHyFP.get2 () > sizez_tfsf - 0.1 && coordHyFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf
          && coordHyFP.get2 () > sizez_tfsf - 0.1 && coordHyFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }

      if (coordHyFP.get1 () > sizex_tfsf - 0.1 && coordHyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHyFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizex_tfsf - 0.1 && coordHyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHyFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordEzFP.get1 () == sizex_tfsf
          && coordEzFP.get2 () > sizez_tfsf + 0.4 && coordEzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf
          && coordEzFP.get2 () > sizez_tfsf + 0.4 && coordEzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordHyFP.get1 () == sizex_tfsf - 0.5
          && coordHyFP.get2 () > sizez_tfsf + 0.4 && coordHyFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHyFP.get2 () > sizez_tfsf + 0.4 && coordHyFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }

      if (coordHyFP.get1 () > sizex_tfsf + 0.4 && coordHyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHyFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizex_tfsf + 0.4 && coordHyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHyFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TEY */

#if defined (MODE_TEZ)

template<LayoutType layout_type>
void testFuncDim2_TEz (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sx, grid_coord sy)
{
#if PRINT_MESSAGE
  printf ("TEz\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  GridCoordinate2D size (sizex, sizey, ct1, ct2);
  GridCoordinate2D sizePML (sx, sy, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizex_tfsf, sizey_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizex_tfsf, sizey_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sx || coordExFP.get1 () >= sizex - sx
        || coordExFP.get2 () < sy || coordExFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordEyFP.get1 () < sx || coordEyFP.get1 () >= sizex - sx
        || coordEyFP.get2 () < sy || coordEyFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordHzFP.get1 () < sx || coordHzFP.get1 () >= sizex - sx
        || coordHzFP.get2 () < sy || coordHzFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR && !doNeedUpdateExB && !doNeedUpdateExF);
    ALWAYS_ASSERT (!doNeedUpdateEyD && !doNeedUpdateEyU && !doNeedUpdateEyB && !doNeedUpdateEyF);
    ALWAYS_ASSERT (!doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordEyFP.get1 () == sizex_tfsf - 0.5
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordHzFP.get1 () == sizex_tfsf
          && coordHzFP.get2 () > sizey_tfsf - 0.1 && coordHzFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf
          && coordHzFP.get2 () > sizey_tfsf - 0.1 && coordHzFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }

      if (coordHzFP.get1 () > sizex_tfsf - 0.1 && coordHzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHzFP.get2 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () > sizex_tfsf - 0.1 && coordHzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHzFP.get2 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordEyFP.get1 () == sizex_tfsf
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordHzFP.get1 () == sizex_tfsf - 0.5
          && coordHzFP.get2 () > sizey_tfsf + 0.4 && coordHzFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHzFP.get2 () > sizey_tfsf + 0.4 && coordHzFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }

      if (coordHzFP.get1 () > sizex_tfsf + 0.4 && coordHzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHzFP.get2 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () > sizex_tfsf + 0.4 && coordHzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHzFP.get2 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TEZ */

#if defined (MODE_TMX)

template<LayoutType layout_type>
void testFuncDim2_TMx (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sy, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("TMx\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate2D size (sizey, sizez, ct1, ct2);
  GridCoordinate2D sizePML (sy, sz, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizey_tfsf, sizez_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizey_tfsf, sizez_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < 1; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sy || coordExFP.get1 () >= sizey - sy
        || coordExFP.get2 () < sz || coordExFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordHyFP.get1 () < sy || coordHyFP.get1 () >= sizey - sy
        || coordHyFP.get2 () < sz || coordHyFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    if (coordHzFP.get1 () < sy || coordHzFP.get1 () >= sizey - sy
        || coordHzFP.get2 () < sz || coordHzFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR);
    ALWAYS_ASSERT (!doNeedUpdateHyL && !doNeedUpdateHyR && !doNeedUpdateHyD && !doNeedUpdateHyU);
    ALWAYS_ASSERT (!doNeedUpdateHzL && !doNeedUpdateHzR && !doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      if (coordExFP.get1 () == sizey_tfsf - 0.5
          && coordExFP.get2 () > sizez_tfsf + 0.4 && coordExFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () == sizey - sizey_tfsf + 0.5
          && coordExFP.get2 () > sizez_tfsf + 0.4 && coordExFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordExFP.get1 () > sizey_tfsf + 0.4 && coordExFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordExFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizey_tfsf + 0.4 && coordExFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordExFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordHyFP.get1 () > sizey_tfsf + 0.4 && coordHyFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizey_tfsf + 0.4 && coordHyFP.get1 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }

      if (coordHzFP.get1 () == sizey_tfsf
          && coordHzFP.get2 () > sizez_tfsf + 0.4 && coordHzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () == sizey - sizey_tfsf
          && coordHzFP.get2 () > sizez_tfsf + 0.4 && coordHzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordExFP.get1 () == sizey_tfsf
          && coordExFP.get2 () > sizez_tfsf - 0.1 && coordExFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () == sizey - sizey_tfsf
          && coordExFP.get2 () > sizez_tfsf - 0.1 && coordExFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordExFP.get1 () > sizey_tfsf - 0.1 && coordExFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordExFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizey_tfsf - 0.1 && coordExFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordExFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      if (coordHyFP.get1 () > sizey_tfsf - 0.1 && coordHyFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizey_tfsf - 0.1 && coordHyFP.get1 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }

      if (coordHzFP.get1 () == sizey_tfsf - 0.5
          && coordHzFP.get2 () > sizez_tfsf - 0.1 && coordHzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () == sizey - sizey_tfsf + 0.5
          && coordHzFP.get2 () > sizez_tfsf - 0.1 && coordHzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TMX */

#if defined (MODE_TMY)

template<LayoutType layout_type>
void testFuncDim2_TMy (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sx, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("TMy\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate2D size (sizex, sizez, ct1, ct2);
  GridCoordinate2D sizePML (sx, sz, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizex_tfsf, sizez_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizex_tfsf, sizez_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < 1; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEyFP.get1 () < sx || coordEyFP.get1 () >= sizex - sx
        || coordEyFP.get2 () < sz || coordEyFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordHxFP.get1 () < sx || coordHxFP.get1 () >= sizex - sx
        || coordHxFP.get2 () < sz || coordHxFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    if (coordHzFP.get1 () < sx || coordHzFP.get1 () >= sizex - sx
        || coordHzFP.get2 () < sz || coordHzFP.get2 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEyD && !doNeedUpdateEyU);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR && !doNeedUpdateHxD && !doNeedUpdateHxU);
    ALWAYS_ASSERT (!doNeedUpdateHzD && !doNeedUpdateHzU && !doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      if (coordEyFP.get1 () == sizex_tfsf - 0.5
          && coordEyFP.get2 () > sizez_tfsf + 0.4 && coordEyFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEyFP.get2 () > sizez_tfsf + 0.4 && coordEyFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordEyFP.get1 () > sizex_tfsf + 0.4 && coordEyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEyFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizex_tfsf + 0.4 && coordEyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEyFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }

      if (coordHzFP.get1 () == sizex_tfsf
          && coordHzFP.get2 () > sizez_tfsf + 0.4 && coordHzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf
          && coordHzFP.get2 () > sizez_tfsf + 0.4 && coordHzFP.get2 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEyFP.get1 () == sizex_tfsf
          && coordEyFP.get2 () > sizez_tfsf - 0.1 && coordEyFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf
          && coordEyFP.get2 () > sizez_tfsf - 0.1 && coordEyFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordEyFP.get1 () > sizex_tfsf - 0.1 && coordEyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEyFP.get2 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizex_tfsf - 0.1 && coordEyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEyFP.get2 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }

      if (coordHzFP.get1 () == sizex_tfsf - 0.5
          && coordHzFP.get2 () > sizez_tfsf - 0.1 && coordHzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHzFP.get2 () > sizez_tfsf - 0.1 && coordHzFP.get2 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TMY */

#if defined (MODE_TMZ)

template<LayoutType layout_type>
void testFuncDim2_TMz (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                       grid_coord mult, grid_coord sx, grid_coord sy)
{
#if PRINT_MESSAGE
  printf ("TEz\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;
  CoordinateType ct3 = CoordinateType::NONE;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  GridCoordinate2D size (sizex, sizey, ct1, ct2);
  GridCoordinate2D sizePML (sx, sy, ct1, ct2);
  GridCoordinate2D sizeTFSFLeft (sizex_tfsf, sizey_tfsf, ct1, ct2);
  GridCoordinate2D sizeTFSFRight (sizex_tfsf, sizey_tfsf, ct1, ct2);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, layout_type> layout (size,
                                                                                     sizePML,
                                                                                     sizeTFSFLeft,
                                                                                     sizeTFSFRight,
                                                                                     incAngle1,
                                                                                     incAngle2,
                                                                                     incAngle3,
                                                                                     doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < 1; ++k)
  {
    #define TCoord GridCoordinate2DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordEzFP.get1 () < sx || coordEzFP.get1 () >= sizex - sx
        || coordEzFP.get2 () < sy || coordEzFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHxFP.get1 () < sx || coordHxFP.get1 () >= sizex - sx
        || coordHxFP.get2 () < sy || coordHxFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    if (coordHyFP.get1 () < sx || coordHyFP.get1 () >= sizex - sx
        || coordHyFP.get2 () < sy || coordHyFP.get2 () >= sizey - sy)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR && !doNeedUpdateHxB && !doNeedUpdateHxF);
    ALWAYS_ASSERT (!doNeedUpdateHyD && !doNeedUpdateHyU && !doNeedUpdateHyB && !doNeedUpdateHyF);

    if (layout_type == E_CENTERED)
    {
      if (coordEzFP.get1 () == sizex_tfsf - 0.5
          && coordEzFP.get2 () > sizey_tfsf + 0.4 && coordEzFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEzFP.get2 () > sizey_tfsf + 0.4 && coordEzFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordEzFP.get1 () > sizex_tfsf + 0.4 && coordEzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEzFP.get2 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () > sizex_tfsf + 0.4 && coordEzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEzFP.get2 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHyFP.get1 () == sizex_tfsf
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      if (coordEzFP.get1 () == sizex_tfsf
          && coordEzFP.get2 () > sizey_tfsf - 0.1 && coordEzFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf
          && coordEzFP.get2 () > sizey_tfsf - 0.1 && coordEzFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordEzFP.get1 () > sizex_tfsf - 0.1 && coordEzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEzFP.get2 () == sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () > sizex_tfsf - 0.1 && coordEzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEzFP.get2 () == sizey - sizey_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizey_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizey - sizey_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHyFP.get1 () == sizex_tfsf - 0.5
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_TMZ */

#if defined (MODE_DIM3)

template<LayoutType layout_type>
void testFuncDim3 (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision,
                   grid_coord mult, grid_coord sx, grid_coord sy, grid_coord sz)
{
#if PRINT_MESSAGE
  printf ("dim3\n");
#endif /* PRINT_MESSAGE */
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;
  CoordinateType ct3 = CoordinateType::Z;

  grid_coord sizex = sx * SIZEMULT;
  grid_coord sizex_tfsf = sx * mult;

  grid_coord sizey = sy * SIZEMULT;
  grid_coord sizey_tfsf = sy * mult;

  grid_coord sizez = sz * SIZEMULT;
  grid_coord sizez_tfsf = sz * mult;

  GridCoordinate3D size (sizex, sizey, sizez, ct1, ct2, ct3);
  GridCoordinate3D sizePML (sx, sy, sz, ct1, ct2, ct3);
  GridCoordinate3D sizeTFSFLeft (sizex_tfsf, sizey_tfsf, sizez_tfsf, ct1, ct2, ct3);
  GridCoordinate3D sizeTFSFRight (sizex_tfsf, sizey_tfsf, sizez_tfsf, ct1, ct2, ct3);

  YeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, layout_type> layout (size,
                                                                                 sizePML,
                                                                                 sizeTFSFLeft,
                                                                                 sizeTFSFRight,
                                                                                 incAngle1,
                                                                                 incAngle2,
                                                                                 incAngle3,
                                                                                 doubleMaterialPrecision);

  for (grid_coord i = 0; i < sizex; ++i)
  for (grid_coord j = 0; j < sizey; ++j)
  for (grid_coord k = 0; k < sizez; ++k)
  {
    #define TCoord GridCoordinate3DTemplate
    INITIALIZATION
    #undef TCoord

    if (coordExFP.get1 () < sx || coordExFP.get1 () >= sizex - sx
        || coordExFP.get2 () < sy || coordExFP.get2 () >= sizey - sy
        || coordExFP.get3 () < sz || coordExFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isExInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isExInPML);
    }

    if (coordEyFP.get1 () < sx || coordEyFP.get1 () >= sizex - sx
        || coordEyFP.get2 () < sy || coordEyFP.get2 () >= sizey - sy
        || coordEyFP.get3 () < sz || coordEyFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEyInPML);
    }

    if (coordEzFP.get1 () < sx || coordEzFP.get1 () >= sizex - sx
        || coordEzFP.get2 () < sy || coordEzFP.get2 () >= sizey - sy
        || coordEzFP.get3 () < sz || coordEzFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isEzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isEzInPML);
    }

    if (coordHxFP.get1 () < sx || coordHxFP.get1 () >= sizex - sx
        || coordHxFP.get2 () < sy || coordHxFP.get2 () >= sizey - sy
        || coordHxFP.get3 () < sz || coordHxFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHxInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHxInPML);
    }

    if (coordHyFP.get1 () < sx || coordHyFP.get1 () >= sizex - sx
        || coordHyFP.get2 () < sy || coordHyFP.get2 () >= sizey - sy
        || coordHyFP.get3 () < sz || coordHyFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHyInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHyInPML);
    }

    if (coordHzFP.get1 () < sx || coordHzFP.get1 () >= sizex - sx
        || coordHzFP.get2 () < sy || coordHzFP.get2 () >= sizey - sy
        || coordHzFP.get3 () < sz || coordHzFP.get3 () >= sizez - sz)
    {
      ALWAYS_ASSERT (isHzInPML);
    }
    else
    {
      ALWAYS_ASSERT (!isHzInPML);
    }

    ALWAYS_ASSERT (!doNeedUpdateExL && !doNeedUpdateExR);
    ALWAYS_ASSERT (!doNeedUpdateEyD && !doNeedUpdateEyU);
    ALWAYS_ASSERT (!doNeedUpdateEzB && !doNeedUpdateEzF);
    ALWAYS_ASSERT (!doNeedUpdateHxL && !doNeedUpdateHxR);
    ALWAYS_ASSERT (!doNeedUpdateHyD && !doNeedUpdateHyU);
    ALWAYS_ASSERT (!doNeedUpdateHzB && !doNeedUpdateHzF);

    if (layout_type == E_CENTERED)
    {
      /*
       * Ex
       */
      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizey_tfsf - 0.5
          && coordExFP.get3 () > sizez_tfsf + 0.4 && coordExFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () == sizey - sizey_tfsf + 0.5
          && coordExFP.get3 () > sizez_tfsf + 0.4 && coordExFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () > sizey_tfsf + 0.4 && coordExFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordExFP.get3 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizex_tfsf - 0.1 && coordExFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordExFP.get2 () > sizey_tfsf + 0.4 && coordExFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordExFP.get3 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      /*
       * Ey
       */
      if (coordEyFP.get1 () == sizex_tfsf - 0.5
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get3 () > sizez_tfsf + 0.4 && coordEyFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get3 () > sizez_tfsf + 0.4 && coordEyFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordEyFP.get1 () > sizex_tfsf + 0.4 && coordEyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get3 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizex_tfsf + 0.4 && coordEyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEyFP.get2 () > sizey_tfsf - 0.1 && coordEyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEyFP.get3 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      /*
       * Ez
       */
      if (coordEzFP.get1 () == sizex_tfsf - 0.5
          && coordEzFP.get2 () > sizey_tfsf + 0.4 && coordEzFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEzFP.get3 () > sizez_tfsf - 0.1 && coordEzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordEzFP.get2 () > sizey_tfsf + 0.4 && coordEzFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEzFP.get3 () > sizez_tfsf - 0.1 && coordEzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordEzFP.get1 () > sizex_tfsf + 0.4 && coordEzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEzFP.get2 () == sizey_tfsf - 0.5
          && coordEzFP.get3 () > sizez_tfsf - 0.1 && coordEzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () > sizex_tfsf + 0.4 && coordEzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordEzFP.get2 () == sizey - sizey_tfsf + 0.5
          && coordEzFP.get3 () > sizez_tfsf - 0.1 && coordEzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      /*
       * Hx
       */
      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizey_tfsf
          && coordHxFP.get3 () > sizez_tfsf - 0.1 && coordHxFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () == sizey - sizey_tfsf
          && coordHxFP.get3 () > sizez_tfsf - 0.1 && coordHxFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () > sizey_tfsf - 0.1 && coordHxFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHxFP.get3 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizex_tfsf + 0.4 && coordHxFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHxFP.get2 () > sizey_tfsf - 0.1 && coordHxFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHxFP.get3 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }

      /*
       * Hy
       */
      if (coordHyFP.get1 () == sizex_tfsf
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get3 () > sizez_tfsf - 0.1 && coordHyFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get3 () > sizez_tfsf - 0.1 && coordHyFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }

      if (coordHyFP.get1 () > sizex_tfsf - 0.1 && coordHyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get3 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizex_tfsf - 0.1 && coordHyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHyFP.get2 () > sizey_tfsf + 0.4 && coordHyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHyFP.get3 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }

      /*
       * Hz
       */
      if (coordHzFP.get1 () == sizex_tfsf
          && coordHzFP.get2 () > sizey_tfsf - 0.1 && coordHzFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHzFP.get3 () > sizez_tfsf + 0.4 && coordHzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf
          && coordHzFP.get2 () > sizey_tfsf - 0.1 && coordHzFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHzFP.get3 () > sizez_tfsf + 0.4 && coordHzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }

      if (coordHzFP.get1 () > sizex_tfsf - 0.1 && coordHzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHzFP.get2 () == sizey_tfsf
          && coordHzFP.get3 () > sizez_tfsf + 0.4 && coordHzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () > sizex_tfsf - 0.1 && coordHzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHzFP.get2 () == sizey - sizey_tfsf
          && coordHzFP.get3 () > sizez_tfsf + 0.4 && coordHzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else if (layout_type == H_CENTERED)
    {
      /*
       * Ex
       */
      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizey_tfsf
          && coordExFP.get3 () > sizez_tfsf - 0.1 && coordExFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateExD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExD);
      }

      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () == sizey - sizey_tfsf
          && coordExFP.get3 () > sizez_tfsf - 0.1 && coordExFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateExU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExU);
      }

      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () > sizey_tfsf - 0.1 && coordExFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordExFP.get3 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExB);
      }

      if (coordExFP.get1 () > sizex_tfsf + 0.4 && coordExFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordExFP.get2 () > sizey_tfsf - 0.1 && coordExFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordExFP.get3 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateExF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateExF);
      }

      /*
       * Ey
       */
      if (coordEyFP.get1 () == sizex_tfsf
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get3 () > sizez_tfsf - 0.1 && coordEyFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyL);
      }

      if (coordEyFP.get1 () == sizex - sizex_tfsf
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get3 () > sizez_tfsf - 0.1 && coordEyFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateEyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyR);
      }

      if (coordEyFP.get1 () > sizex_tfsf - 0.1 && coordEyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get3 () == sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyB);
      }

      if (coordEyFP.get1 () > sizex_tfsf - 0.1 && coordEyFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEyFP.get2 () > sizey_tfsf + 0.4 && coordEyFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordEyFP.get3 () == sizez - sizez_tfsf)
      {
        ALWAYS_ASSERT (doNeedUpdateEyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEyF);
      }

      /*
       * Ez
       */
      if (coordEzFP.get1 () == sizex_tfsf
          && coordEzFP.get2 () > sizey_tfsf - 0.1 && coordEzFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEzFP.get3 () > sizez_tfsf + 0.4 && coordEzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzL);
      }

      if (coordEzFP.get1 () == sizex - sizex_tfsf
          && coordEzFP.get2 () > sizey_tfsf - 0.1 && coordEzFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordEzFP.get3 () > sizez_tfsf + 0.4 && coordEzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzR);
      }

      if (coordEzFP.get1 () > sizex_tfsf - 0.1 && coordEzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEzFP.get2 () == sizey_tfsf
          && coordEzFP.get3 () > sizez_tfsf + 0.4 && coordEzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzD);
      }

      if (coordEzFP.get1 () > sizex_tfsf - 0.1 && coordEzFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordEzFP.get2 () == sizey - sizey_tfsf
          && coordEzFP.get3 () > sizez_tfsf + 0.4 && coordEzFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateEzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateEzU);
      }

      /*
       * Hx
       */
      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizey_tfsf - 0.5
          && coordHxFP.get3 () > sizez_tfsf + 0.4 && coordHxFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHxD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxD);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () == sizey - sizey_tfsf + 0.5
          && coordHxFP.get3 () > sizez_tfsf + 0.4 && coordHxFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHxU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxU);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () > sizey_tfsf + 0.4 && coordHxFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHxFP.get3 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxB);
      }

      if (coordHxFP.get1 () > sizex_tfsf - 0.1 && coordHxFP.get1 () < sizex - sizex_tfsf + 0.1
          && coordHxFP.get2 () > sizey_tfsf + 0.4 && coordHxFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHxFP.get3 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHxF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHxF);
      }

      /*
       * Hy
       */
      if (coordHyFP.get1 () == sizex_tfsf - 0.5
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get3 () > sizez_tfsf + 0.4 && coordHyFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyL);
      }

      if (coordHyFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get3 () > sizez_tfsf + 0.4 && coordHyFP.get3 () < sizez - sizez_tfsf - 0.4)
      {
        ALWAYS_ASSERT (doNeedUpdateHyR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyR);
      }

      if (coordHyFP.get1 () > sizex_tfsf + 0.4 && coordHyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get3 () == sizez_tfsf - 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyB);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyB);
      }

      if (coordHyFP.get1 () > sizex_tfsf + 0.4 && coordHyFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHyFP.get2 () > sizey_tfsf - 0.1 && coordHyFP.get2 () < sizey - sizey_tfsf + 0.1
          && coordHyFP.get3 () == sizez - sizez_tfsf + 0.5)
      {
        ALWAYS_ASSERT (doNeedUpdateHyF);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHyF);
      }

      /*
       * Hz
       */
      if (coordHzFP.get1 () == sizex_tfsf - 0.5
          && coordHzFP.get2 () > sizey_tfsf + 0.4 && coordHzFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHzFP.get3 () > sizez_tfsf - 0.1 && coordHzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzL);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzL);
      }

      if (coordHzFP.get1 () == sizex - sizex_tfsf + 0.5
          && coordHzFP.get2 () > sizey_tfsf + 0.4 && coordHzFP.get2 () < sizey - sizey_tfsf - 0.4
          && coordHzFP.get3 () > sizez_tfsf - 0.1 && coordHzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzR);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzR);
      }

      if (coordHzFP.get1 () > sizex_tfsf + 0.4 && coordHzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHzFP.get2 () == sizey_tfsf - 0.5
          && coordHzFP.get3 () > sizez_tfsf - 0.1 && coordHzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzD);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzD);
      }

      if (coordHzFP.get1 () > sizex_tfsf + 0.4 && coordHzFP.get1 () < sizex - sizex_tfsf - 0.4
          && coordHzFP.get2 () == sizey - sizey_tfsf + 0.5
          && coordHzFP.get3 () > sizez_tfsf - 0.1 && coordHzFP.get3 () < sizez - sizez_tfsf + 0.1)
      {
        ALWAYS_ASSERT (doNeedUpdateHzU);
      }
      else
      {
        ALWAYS_ASSERT (!doNeedUpdateHzU);
      }
    }
    else
    {
      UNREACHABLE;
    }
  }
}

#endif /* MODE_DIM3 */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void testFunc (FPValue incAngle1, FPValue incAngle2, FPValue incAngle3, bool doubleMaterialPrecision)
{
  for (grid_coord mult = 2; mult <= MULT; ++mult)
  {
    for (grid_coord sx = SIZEX; sx <= 2*SIZEX; sx += SIZEX)
    {
      for (grid_coord sy = SIZEY; sy <= 2*SIZEY; sy += SIZEY)
      {
        for (grid_coord sz = SIZEZ; sz <= 2*SIZEZ; sz += SIZEZ)
        {
          testFuncInternal<Type, TCoord, layout_type> (incAngle1, incAngle2, incAngle3, doubleMaterialPrecision, mult, sx, sy, sz);
        }
      }
    }
  }
}

int main (int argc, char** argv)
{
  for (int dMaterialPrecision = 0; dMaterialPrecision <= 1; ++dMaterialPrecision)
  {
#if defined (MODE_EX_HY)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED> (0, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED> (0, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_EX_HY */
#if defined (MODE_EX_HZ)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, 0, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, 0, dMaterialPrecision);
#endif /* MODE_EX_HZ */
#if defined (MODE_EY_HX)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED> (0, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED> (0, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_EY_HX */
#if defined (MODE_EY_HZ)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, 0, 0, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, 0, 0, dMaterialPrecision);
#endif /* MODE_EY_HZ */
#if defined (MODE_EZ_HX)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_EZ_HX */
#if defined (MODE_EZ_HY)
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
    testFunc<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_EZ_HY */

    for (grid_coord mult = 2; mult <= MULT; ++mult)
    {
      for (grid_coord sz = SIZEZ; sz <= 2*SIZEZ; sz += SIZEZ)
      {
#if defined (MODE_EX_HY)
        testFuncDim1_ExHy<E_CENTERED> (0, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sz);
        testFuncDim1_ExHy<H_CENTERED> (0, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sz);
#endif /* MODE_EX_HY */
#if defined (MODE_EY_HX)
        testFuncDim1_EyHx<E_CENTERED> (0, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sz);
        testFuncDim1_EyHx<H_CENTERED> (0, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sz);
#endif /* MODE_EY_HX */
      }

      for (grid_coord sy = SIZEY; sy <= 2*SIZEY; sy += SIZEY)
      {
#if defined (MODE_EX_HZ)
        testFuncDim1_ExHz<E_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, 0, dMaterialPrecision, mult, sy);
        testFuncDim1_ExHz<H_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, 0, dMaterialPrecision, mult, sy);
#endif /* MODE_EX_HZ */
#if defined (MODE_EZ_HX)
        testFuncDim1_EzHx<E_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sy);
        testFuncDim1_EzHx<H_CENTERED> (PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sy);
#endif /* MODE_EZ_HX */
      }

      for (grid_coord sx = SIZEX; sx <= 2*SIZEX; sx += SIZEX)
      {
#if defined (MODE_EY_HZ)
        testFuncDim1_EyHz<E_CENTERED> (PhysicsConst::Pi / 2, 0, 0, dMaterialPrecision, mult, sx);
        testFuncDim1_EyHz<H_CENTERED> (PhysicsConst::Pi / 2, 0, 0, dMaterialPrecision, mult, sx);
#endif /* MODE_EY_HZ */
#if defined (MODE_EZ_HY)
        testFuncDim1_EzHy<E_CENTERED> (PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx);
        testFuncDim1_EzHy<H_CENTERED> (PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx);
#endif /* MODE_EZ_HY */
      }
    }

    for (FPValue angle1 = 0.0; angle1 <= PhysicsConst::Pi / 2; angle1 += PhysicsConst::Pi / 4)
    {
#if defined (MODE_TEX)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (angle1, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED> (angle1, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_TEX */
#if defined (MODE_TEY)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (angle1, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED> (angle1, 0, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_TEY */
#if defined (MODE_TEZ)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, angle1, 0, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, angle1, 0, dMaterialPrecision);
#endif /* MODE_TEZ */
#if defined (MODE_TMX)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (angle1, PhysicsConst::Pi / 2, 0, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED> (angle1, PhysicsConst::Pi / 2, 0, dMaterialPrecision);
#endif /* MODE_TMX */
#if defined (MODE_TMY)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (angle1, 0, 0, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED> (angle1, 0, 0, dMaterialPrecision);
#endif /* MODE_TMY */
#if defined (MODE_TMZ)
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (PhysicsConst::Pi / 2, angle1, PhysicsConst::Pi / 2, dMaterialPrecision);
      testFunc<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED> (PhysicsConst::Pi / 2, angle1, PhysicsConst::Pi / 2, dMaterialPrecision);
#endif /* MODE_TMZ */

      for (grid_coord mult = 2; mult <= MULT; ++mult)
      {
        for (grid_coord sy = SIZEY; sy <= 2*SIZEY; sy += SIZEY)
        for (grid_coord sz = SIZEZ; sz <= 2*SIZEZ; sz += SIZEZ)
        {
#if defined (MODE_TEX)
          testFuncDim2_TEx<E_CENTERED> (angle1, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sy, sz);
          testFuncDim2_TEx<H_CENTERED> (angle1, PhysicsConst::Pi / 2, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sy, sz);
#endif /* MODE_TEX */
#if defined (MODE_TMX)
          testFuncDim2_TMx<E_CENTERED> (angle1, PhysicsConst::Pi / 2, 0, dMaterialPrecision, mult, sy, sz);
          testFuncDim2_TMx<H_CENTERED> (angle1, PhysicsConst::Pi / 2, 0, dMaterialPrecision, mult, sy, sz);
#endif /* MODE_TMX */
        }

        for (grid_coord sx = SIZEX; sx <= 2*SIZEX; sx += SIZEX)
        for (grid_coord sz = SIZEZ; sz <= 2*SIZEZ; sz += SIZEZ)
        {
#if defined (MODE_TEY)
          testFuncDim2_TEy<E_CENTERED> (angle1, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx, sz);
          testFuncDim2_TEy<H_CENTERED> (angle1, 0, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx, sz);
#endif /* MODE_TEY */
#if defined (MODE_TMY)
          testFuncDim2_TMy<E_CENTERED> (angle1, 0, 0, dMaterialPrecision, mult, sx, sz);
          testFuncDim2_TMy<H_CENTERED> (angle1, 0, 0, dMaterialPrecision, mult, sx, sz);
#endif /* MODE_TMY */
        }

        for (grid_coord sx = SIZEX; sx <= 2*SIZEX; sx += SIZEX)
        for (grid_coord sy = SIZEY; sy <= 2*SIZEY; sy += SIZEY)
        {
#if defined (MODE_TEZ)
          testFuncDim2_TEz<E_CENTERED> (PhysicsConst::Pi / 2, angle1, 0, dMaterialPrecision, mult, sx, sy);
          testFuncDim2_TEz<H_CENTERED> (PhysicsConst::Pi / 2, angle1, 0, dMaterialPrecision, mult, sx, sy);
#endif /* MODE_TEZ */
#if defined (MODE_TMZ)
          testFuncDim2_TMz<E_CENTERED> (PhysicsConst::Pi / 2, angle1, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx, sy);
          testFuncDim2_TMz<H_CENTERED> (PhysicsConst::Pi / 2, angle1, PhysicsConst::Pi / 2, dMaterialPrecision, mult, sx, sy);
#endif /* MODE_TMZ */
        }
      }

      for (FPValue angle2 = 0.0; angle2 <= PhysicsConst::Pi / 2; angle2 += PhysicsConst::Pi / 4)
      {
        for (FPValue angle3 = 0.0; angle3 <= PhysicsConst::Pi / 2; angle3 += PhysicsConst::Pi / 4)
        {
#if defined (MODE_DIM3)
          testFunc<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (angle1, angle2, angle3, dMaterialPrecision);
          testFunc<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED> (angle1, angle2, angle3, dMaterialPrecision);
#endif /* MODE_DIM3 */

          for (grid_coord mult = 2; mult <= MULT; ++mult)
          {
            for (grid_coord sx = SIZEX; sx <= 2*SIZEX; sx += SIZEX)
            for (grid_coord sy = SIZEY; sy <= 2*SIZEY; sy += SIZEY)
            for (grid_coord sz = SIZEZ; sz <= 2*SIZEZ; sz += SIZEZ)
            {
#if defined (MODE_DIM3)
              testFuncDim3<E_CENTERED> (angle1, angle2, angle3, dMaterialPrecision, mult, sx, sy, sz);
              testFuncDim3<H_CENTERED> (angle1, angle2, angle3, dMaterialPrecision, mult, sx, sy, sz);
#endif /* MODE_DIM3 */
            }
          }
        }
      }
    }
  }

  return 0;
} /* main */

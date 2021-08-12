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

#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMz */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
template <>
IDEVICE void
ISCHEME<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
#endif /* MODE_DIM3 */

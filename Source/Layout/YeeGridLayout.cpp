#include "YeeGridLayout.h"

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
const bool YeeGridLayout<Type, TCoord, layout_type>::isParallel = false;

/*
 * Ex
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getExCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getExCoordFP (coord), rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getExCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getExCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}

template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, -5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, -5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, -5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, -5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 5> (getExCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}

template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateExBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getExCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-5> (getExCoordFP (coord).get2 (), leftBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getExCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getExCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<5> (getExCoordFP (coord).get2 (), rightBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getExCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getExCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getExCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-5> (getExCoordFP (coord).get3 (), leftBorderTotalFieldFP.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getExCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getExCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<5> (getExCoordFP (coord).get3 (), rightBorderTotalFieldFP.get3 ()));
}

/*
 * Ey
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getEyCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getEyCoordFP (coord), rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getEyCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getEyCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}

template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, -5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, -5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, -5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, -5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 5> (getEyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEyBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder<-5> (getEyCoordFP (coord).get1 (), leftBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getEyCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<5> (getEyCoordFP (coord).get1 (), rightBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getEyCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getEyCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-5> (getEyCoordFP (coord).get3 (), leftBorderTotalFieldFP.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getEyCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<5> (getEyCoordFP (coord).get3 (), rightBorderTotalFieldFP.get3 ()));
}

/*
 * Ez
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getEzCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getEzCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<-5> (getEzCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<5> (getEzCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, -5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, -5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, -5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, -5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 5> (getEzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateEzBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder<-5> (getEzCoordFP (coord).get1 (), leftBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getEzCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<5> (getEzCoordFP (coord).get1 (), rightBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getEzCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getEzCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-5> (getEzCoordFP (coord).get2 (), leftBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getEzCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<5> (getEzCoordFP (coord).get2 (), rightBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getEzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 false,
                                 false);
}

/*
 * Hx
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHxCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHxCoordFP (coord), rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHxCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHxCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 0> (getHxCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHxBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getHxCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHxCoordFP (coord).get2 (), leftBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHxCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getHxCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHxCoordFP (coord).get2 (), rightBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHxCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getHxCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHxCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHxCoordFP (coord).get3 (), leftBorderTotalFieldFP.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<4, -4> (getHxCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHxCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHxCoordFP (coord).get3 (), rightBorderTotalFieldFP.get3 ()));
}

/*
 * Hy
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHyCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHyCoordFP (coord), rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHyCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHyCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<4, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<4, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP));
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 0> (getHyCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHyBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder<0> (getHyCoordFP (coord).get1 (), leftBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHyCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<0> (getHyCoordFP (coord).get1 (), rightBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHyCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHyCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHyCoordFP (coord).get3 (), leftBorderTotalFieldFP.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHyCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHyCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHyCoordFP (coord).get3 (), rightBorderTotalFieldFP.get3 ()));
}

/*
 * Hz
 */
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHzCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHzCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder1DFirst__1<0> (getHzCoordFP (coord), leftBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder1DSecond__1<0> (getHzCoordFP (coord), rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate1D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<-1, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<-1, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__1<-1, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__1<-1, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 false,
                                 false,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder2DFirst__2<4, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 YeeGridLayoutHelper::tfsfBorder2DSecond__2<4, 0> (getHzCoordFP (coord), leftBorderTotalFieldFP, rightBorderTotalFieldFP),
                                 false,
                                 false,
                                 false,
                                 false);
}
template <>
bool
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate2D coord, LayoutDirection dir) const
{
  return false;
}
template <>
bool
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::doNeedTFSFUpdateHzBorder (GridCoordinate3D coord, LayoutDirection dir) const
{
  return YeeGridLayoutHelper::doNeedTFSFUpdateBorder (dir,
                                 YeeGridLayoutHelper::tfsfBorder<0> (getHzCoordFP (coord).get1 (), leftBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHzCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<0> (getHzCoordFP (coord).get1 (), rightBorderTotalFieldFP.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHzCoordFP (coord).get2 (), leftBorderTotalField.get2 (), rightBorderTotalField.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHzCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHzCoordFP (coord).get2 (), leftBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 YeeGridLayoutHelper::tfsfBorder<-1, 1> (getHzCoordFP (coord).get1 (), leftBorderTotalField.get1 (), rightBorderTotalField.get1 ())
                                 && YeeGridLayoutHelper::tfsfBorder<0> (getHzCoordFP (coord).get2 (), rightBorderTotalFieldFP.get2 ())
                                 && YeeGridLayoutHelper::tfsfBorder<4, -4> (getHzCoordFP (coord).get3 (), leftBorderTotalField.get3 (), rightBorderTotalField.get3 ()),
                                 false,
                                 false);
}

template <>
bool YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP1D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML1D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP2D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML2D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}
template <>
bool YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::isInPML (GridCoordinateFP3D realCoordFP) const
{
  return YeeGridLayoutHelper::isInPML3D (realCoordFP, zeroCoordFP, leftBorderPML, rightBorderPML);
}

#define DEFAULT_VALS_LIST \
  zeroCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minEpsCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minMuCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minExCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minEyCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minEzCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minHxCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minHyCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minHzCoord (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3)) \
  , minEpsCoordFP (TCFP::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)) \
  , minMuCoordFP (TCFP::initAxesCoordinate (0.5, 0.5, 0.5, ct1, ct2, ct3)) \
  , zeroCoordFP (TCFP::initAxesCoordinate (0.0, 0.0, 0.0, ct1, ct2, ct3)) \
  , minExCoordFP (TCFP::initAxesCoordinate (1.0, 0.5, 0.5, ct1, ct2, ct3)) \
  , minEyCoordFP (TCFP::initAxesCoordinate (0.5, 1.0, 0.5, ct1, ct2, ct3)) \
  , minEzCoordFP (TCFP::initAxesCoordinate (0.5, 0.5, 1.0, ct1, ct2, ct3)) \
  , minHxCoordFP (TCFP::initAxesCoordinate (0.5, 1.0, 1.0, ct1, ct2, ct3)) \
  , minHyCoordFP (TCFP::initAxesCoordinate (1.0, 0.5, 1.0, ct1, ct2, ct3)) \
  , minHzCoordFP (TCFP::initAxesCoordinate (1.0, 1.0, 0.5, ct1, ct2, ct3)) \
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
  , leftBorderTotalFieldFP (zeroCoordFP + convertCoord (leftBorderTotalField)) \
  , rightBorderTotalFieldFP (zeroCoordFP + convertCoord (rightBorderTotalField)) \
  , incidentWaveAngle1 (incWaveAngle1) \
  , incidentWaveAngle2 (incWaveAngle2) \
  , incidentWaveAngle3 (incWaveAngle3) \
  , isDoubleMaterialPrecision (doubleMaterialPrecision)

template <>
YeeGridLayout<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Z)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == incWaveAngle2 == incWaveAngle3 == 0);
}

template <>
YeeGridLayout<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Y)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle2 == PhysicsConst::Pi / 2 && incWaveAngle3 == 0);
}

template <>
YeeGridLayout<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Z)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == incWaveAngle2 == incWaveAngle3 == 0);
}

template <>
YeeGridLayout<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle2 == 0 && incWaveAngle3 == 0);
}

template <>
YeeGridLayout<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Y)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle2 == PhysicsConst::Pi / 2 && incWaveAngle3 == PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate1D coordSize,
   GridCoordinate1D sizePML,
   GridCoordinate1D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::NONE)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5, ct1)
{
  ASSERT (size.get1 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle2 == 0 && incWaveAngle3 == PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Y)
, ct2 (CoordinateType::Z)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * sin (incWaveAngle1),
                  leftBorderTotalField.get2 () - 2.5 * cos (incWaveAngle1), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle2 == PhysicsConst::Pi / 2 && incWaveAngle3 == PhysicsConst::Pi / 2);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::Z)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * sin (incWaveAngle1),
                  leftBorderTotalField.get2 () - 2.5 * cos (incWaveAngle1), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle2 == 0 && incWaveAngle3 == PhysicsConst::Pi / 2);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::Y)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * cos (incWaveAngle2),
                  leftBorderTotalField.get2 () - 2.5 * sin (incWaveAngle2), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle3 == 0);

  // TODO: add other angles
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::Y)
, ct2 (CoordinateType::Z)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * sin (incWaveAngle1),
                  leftBorderTotalField.get2 () - 2.5 * cos (incWaveAngle1), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle2 == PhysicsConst::Pi / 2 && incWaveAngle3 == 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::Z)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * sin (incWaveAngle1),
                  leftBorderTotalField.get2 () - 2.5 * cos (incWaveAngle1), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle2 == 0 && incWaveAngle3 == 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate2D coordSize,
   GridCoordinate2D sizePML,
   GridCoordinate2D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::Y)
, ct3 (CoordinateType::NONE)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * cos (incWaveAngle2),
                  leftBorderTotalField.get2 () - 2.5 * sin (incWaveAngle2), ct1, ct2)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0);
  ASSERT (incWaveAngle1 == PhysicsConst::Pi / 2 && incWaveAngle3 == PhysicsConst::Pi / 2);

  // TODO: add other angles
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
}

template <>
YeeGridLayout<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED>::YeeGridLayout
  (GridCoordinate3D coordSize,
   GridCoordinate3D sizePML,
   GridCoordinate3D sizeScatteredZone,
   FPValue incWaveAngle1, /**< teta */
   FPValue incWaveAngle2, /**< phi */
   FPValue incWaveAngle3, /**< psi */
   bool doubleMaterialPrecision)
: ct1 (CoordinateType::X)
, ct2 (CoordinateType::Y)
, ct3 (CoordinateType::Z)
, DEFAULT_VALS_LIST
// TODO: check this, only one angle should be used!
, zeroIncCoordFP (leftBorderTotalField.get1 () - 2.5 * sin (incWaveAngle1) * cos (incWaveAngle2),
                  leftBorderTotalField.get2 () - 2.5 * sin (incWaveAngle1) * sin (incWaveAngle2),
                  leftBorderTotalField.get3 () - 2.5 * cos (incWaveAngle1), ct1, ct2, ct3)
{
  ASSERT (size.get1 () > 0 && size.get2 () > 0 && size.get3 () > 0);

  // TODO: add other angles
  ASSERT (incWaveAngle1 >= 0 && incWaveAngle1 <= PhysicsConst::Pi / 2);
  ASSERT (incWaveAngle2 >= 0 && incWaveAngle2 <= PhysicsConst::Pi / 2);
  ASSERT (incWaveAngle3 >= 0 && incWaveAngle3 <= PhysicsConst::Pi / 2);
}

#undef DEFAULT_VALS_LIST

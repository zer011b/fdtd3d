#define CUDA_SOURCES

#ifdef CUDA_ENABLED

#include "InternalScheme.cpp"

SPECIALIZE_TEMPLATE(InternalSchemeGPU, InternalSchemeHelperGPU,
                    CUDA_DEVICE bool, CUDA_DEVICE bool, CUDA_DEVICE bool,
                    doSkipBorderFunc,
                    (GridCoordinate1D pos, CudaGrid<GridCoordinate1D> *grid),
                    (GridCoordinate2D pos, CudaGrid<GridCoordinate2D> *grid),
                    (GridCoordinate3D pos, CudaGrid<GridCoordinate3D> *grid),
                    (pos, grid))

#ifdef ENABLE_ASSERTS

#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFExAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
//   ASSERT (pos11.get3 () == pos12.get3 ());
//   ASSERT (pos21.get3 () < pos22.get3 ());
// }
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFEyAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
//   ASSERT (pos11.get3 () < pos12.get3 ());
//   ASSERT (pos21.get3 () == pos22.get3 ());
// }
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFEzAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
//   ASSERT (pos11.get3 () == pos12.get3 ());
//   ASSERT (pos21.get3 () == pos22.get3 ());
// }
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHxAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
//   ASSERT (pos11.get3 () < pos12.get3 ());
//   ASSERT (pos21.get3 () == pos22.get3 ());
// }
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () < pos22.get2 ());
// }
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHyAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
//   ASSERT (pos11.get3 () == pos12.get3 ());
//   ASSERT (pos21.get3 () < pos22.get3 ());
// }
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
// }
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
// }
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () < pos12.get1 ());
//   ASSERT (pos21.get1 () == pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () == pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
// }
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  UNREACHABLE;
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE void
InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}
// template <>
// CUDA_DEVICE void
// InternalSchemeGPU<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED>::calculateTFSFHzAsserts
//   (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
// {
//   ASSERT (pos11.get1 () == pos12.get1 ());
//   ASSERT (pos21.get1 () < pos22.get1 ());
//   ASSERT (pos11.get2 () < pos12.get2 ());
//   ASSERT (pos21.get2 () == pos22.get2 ());
//   ASSERT (pos11.get3 () == pos12.get3 ());
//   ASSERT (pos21.get3 () == pos22.get3 ());
// }
#endif /* MODE_DIM3 */

#endif /* ENABLE_ASSERTS */


#ifdef MODE_EX_HY
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == 0);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
CUDA_DEVICE
FieldValue
InternalSchemeHelperGPU::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (GridCoordinateFP3D realCoord, GridCoordinateFP3D zeroCoordFP,
                                                   FPValue dDiff, CudaGrid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  FPValue z = realCoord.get3 () - zeroCoordFP.get3 ();

  FPValue d = x * sin (incAngle1) * cos (incAngle2) + y * sin (incAngle1) * sin (incAngle2) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelperGPU::approximateIncidentWaveHelper (d, FieldInc);
}
#endif /* MODE_DIM3 */

#endif /* CUDA_ENABLED */

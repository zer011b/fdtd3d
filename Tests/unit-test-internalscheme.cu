/*
 * Unit test for InternalScheme on GPU
 */

#define CUDA_SOURCES

#include <iostream>

#include "InternalScheme.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

#define SIZE 20
#define PML_SIZE 5
#define TFSF_SIZE 7

#define LAMBDA 0.2
#define DX 0.02

#define TIMESTEPS 100

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
void test (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *intScheme)
{
  TCoord<grid_coord, true> diff = intScheme->getEps ()->getBufSize ();
  TCoord<grid_coord, true> overallSize = intScheme->getEps ()->getSize ();

  for (time_step t = 0; t < SOLVER_SETTINGS.getNumTimeSteps (); ++t)
  {
    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      intScheme->performPlaneWaveESteps (t);
    }

    if (intScheme->getDoNeedEx ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, diff, overallSize - diff);
      intScheme->getEx ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEy ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, diff, overallSize - diff);
      intScheme->getEy ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEz ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, diff, overallSize - diff);
      intScheme->getEz ()->nextTimeStep ();
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      intScheme->performPlaneWaveHSteps (t);
    }

    if (intScheme->getDoNeedHx ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, diff, overallSize - diff);
      intScheme->getHx ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHy ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, diff, overallSize - diff);
      intScheme->getHy ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHz ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, diff, overallSize - diff);
      intScheme->getHz ()->nextTimeStep ();
    }
  }
}

template <template <typename> class TGrid>
void test1D (TGrid<GridCoordinate1D> *E,
             GridCoordinateFP1D diff,
             CoordinateType ct1)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      GridCoordinate1D pos (i, ct1);
      GridCoordinateFP1D posFP (i, ct1);
      posFP = posFP + diff;

      FieldPointValue * val = E->getFieldPointValue (pos);
      FieldValue cur = val->getCurValue ();

      if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE)
      {
        ASSERT (SQR (cur.abs () - FPValue (1)) < 0.0001);
      }
      else
      {
        ASSERT (IS_FP_EXACT (cur.abs (), FPValue (0)));
      }
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

template <template <typename> class TGrid>
void test2D (TGrid<GridCoordinate2D> *E,
             GridCoordinateFP2D diff,
             CoordinateType ct1,
             CoordinateType ct2)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      for (grid_coord j = 0; j < SIZE; ++j)
      {
        GridCoordinate2D pos (i, j, ct1, ct2);
        GridCoordinateFP2D posFP (i, j, ct1, ct2);
        posFP = posFP + diff;

        FieldPointValue * val = E->getFieldPointValue (pos);
        FieldValue cur = val->getCurValue ();

        if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
            && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE)
        {
          ASSERT (SQR (cur.abs () - FPValue (1)) < 0.0001);
        }
        else
        {
          ASSERT (IS_FP_EXACT (cur.abs (), FPValue (0)));
        }
      }
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

template <template <typename> class TGrid>
void test3D (TGrid<GridCoordinate3D> *E,
             GridCoordinateFP3D diff,
             CoordinateType ct1,
             CoordinateType ct2,
             CoordinateType ct3)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      for (grid_coord j = 0; j < SIZE; ++j)
      {
        for (grid_coord k = 0; k < SIZE; ++k)
        {
          GridCoordinate3D pos (i, j, k, ct1, ct2, ct3);
          GridCoordinateFP3D posFP (i, j, k, ct1, ct2, ct3);
          posFP = posFP + diff;

          FieldPointValue * val = E->getFieldPointValue (pos);
          FieldValue cur = val->getCurValue ();

          if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
              && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE
              && posFP.get3 () >= TFSF_SIZE && posFP.get3 () <= SIZE - TFSF_SIZE)
          {
            ASSERT (SQR (cur.abs () - FPValue (1)) < 0.0001);
          }
          else
          {
            ASSERT (IS_FP_EXACT (cur.abs (), FPValue (0)));
          }
        }
      }
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

__global__ void launch1D_ExHy (CudaExitStatus *retval,
                               InternalScheme1D_ExHy_CudaGrid<E_CENTERED> *intScheme)
{
  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED, CudaGrid> (intScheme);

  *retval = CUDA_OK;
}

template<LayoutType layout_type>
void test1D_ExHy ()
{
  CoordinateType ct1 = CoordinateType::Z;

  GridCoordinate1D overallSize (SIZE, ct1);
  GridCoordinate1D pmlSize (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 0;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme1D_ExHy_Grid<layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  intScheme.getEps ()->initialize (getFieldValueRealOnly (1.0));
  intScheme.getMu ()->initialize (getFieldValueRealOnly (1.0));

  GridCoordinate1D zero (0, ct1);

  /*
   * Init InternalScheme on GPU
   */
  InternalScheme1D_ExHy_CudaGrid<layout_type> gpuScheme;
  gpuScheme.initFromCPU (&intScheme, zero, overallSize);

  InternalScheme1D_ExHy_CudaGrid<layout_type> tmpGPUScheme;
  tmpGPUScheme.initOnGPU ();

  InternalScheme1D_ExHy_CudaGrid<layout_type> *d_gpuScheme;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuScheme, sizeof(InternalScheme1D_ExHy_CudaGrid<layout_type>)));

  /*
   * Copy InternalScheme to GPU
   */
  gpuScheme.copyFromCPU (zero, overallSize);

  tmpGPUScheme.copyToGPU (&gpuScheme);

  cudaCheckErrorCmd (cudaMemcpy (d_gpuScheme, &tmpGPUScheme, sizeof(InternalScheme1D_ExHy_CudaGrid<layout_type>), cudaMemcpyHostToDevice));

  /*
   * d_gpuScheme is now fully initialized to be used on GPU
   */
  CudaExitStatus _retval = CUDA_ERROR;
  CudaExitStatus *retval = &_retval;
  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocks (overallSize.get1 () / 4, 1, 1);
  dim3 threads (4, 1, 1);

  cudaCheckExitStatus (launch1D_ExHy <<< blocks, threads >>> (exitStatusCuda, d_gpuScheme));



  // test1D<Grid> (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1);
}
//
// template<LayoutType layout_type>
// void test1D_ExHz ()
// {
//   CoordinateType ct1 = CoordinateType::Y;
//
//   GridCoordinate1D overallSize (SIZE, ct1);
//   GridCoordinate1D pmlSize (PML_SIZE, ct1);
//   GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
//   GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 0;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
//     (overallSize,
//      pmlSize,
//      tfsfSizeLeft,
//      tfsfSizeRight,
//      angle1 * PhysicsConst::Pi / 180.0,
//      angle2 * PhysicsConst::Pi / 180.0,
//      angle3 * PhysicsConst::Pi / 180.0,
//      useDoubleMaterialPrecision);
//
//   InternalScheme1D_ExHz_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);
//
//   test1D<Grid> (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1);
// }
//
// template<LayoutType layout_type>
// void test1D_EyHx ()
// {
//   CoordinateType ct1 = CoordinateType::Z;
//
//   GridCoordinate1D overallSize (SIZE, ct1);
//   GridCoordinate1D pmlSize (PML_SIZE, ct1);
//   GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
//   GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 0;
//   FPValue angle2 = 90;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
//     (overallSize,
//      pmlSize,
//      tfsfSizeLeft,
//      tfsfSizeRight,
//      angle1 * PhysicsConst::Pi / 180.0,
//      angle2 * PhysicsConst::Pi / 180.0,
//      angle3 * PhysicsConst::Pi / 180.0,
//      useDoubleMaterialPrecision);
//
//   InternalScheme1D_EyHx_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);
//
//   test1D<Grid> (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1);
// }
//
// template<LayoutType layout_type>
// void test1D_EyHz ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//
//   GridCoordinate1D overallSize (SIZE, ct1);
//   GridCoordinate1D pmlSize (PML_SIZE, ct1);
//   GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
//   GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 0;
//   FPValue angle3 = 0;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
//     (overallSize,
//      pmlSize,
//      tfsfSizeLeft,
//      tfsfSizeRight,
//      angle1 * PhysicsConst::Pi / 180.0,
//      angle2 * PhysicsConst::Pi / 180.0,
//      angle3 * PhysicsConst::Pi / 180.0,
//      useDoubleMaterialPrecision);
//
//   InternalScheme1D_EyHz_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);
//
//   test1D<Grid> (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1);
// }
//
// template<LayoutType layout_type>
// void test1D_EzHx ()
// {
//   CoordinateType ct1 = CoordinateType::Y;
//
//   GridCoordinate1D overallSize (SIZE, ct1);
//   GridCoordinate1D pmlSize (PML_SIZE, ct1);
//   GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
//   GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
//     (overallSize,
//      pmlSize,
//      tfsfSizeLeft,
//      tfsfSizeRight,
//      angle1 * PhysicsConst::Pi / 180.0,
//      angle2 * PhysicsConst::Pi / 180.0,
//      angle3 * PhysicsConst::Pi / 180.0,
//      useDoubleMaterialPrecision);
//
//   InternalScheme1D_EzHx_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);
//
//   test1D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1);
// }
//
// template<LayoutType layout_type>
// void test1D_EzHy ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//
//   GridCoordinate1D overallSize (SIZE, ct1);
//   GridCoordinate1D pmlSize (PML_SIZE, ct1);
//   GridCoordinate1D tfsfSizeLeft (TFSF_SIZE, ct1);
//   GridCoordinate1D tfsfSizeRight (TFSF_SIZE, ct1);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 0;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> yeeLayout
//     (overallSize,
//      pmlSize,
//      tfsfSizeLeft,
//      tfsfSizeRight,
//      angle1 * PhysicsConst::Pi / 180.0,
//      angle2 * PhysicsConst::Pi / 180.0,
//      angle3 * PhysicsConst::Pi / 180.0,
//      useDoubleMaterialPrecision);
//
//   InternalScheme1D_EzHy_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);
//
//   test1D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1);
// }
//
// template<LayoutType layout_type>
// void test2D_TEx ()
// {
//   CoordinateType ct1 = CoordinateType::Y;
//   CoordinateType ct2 = CoordinateType::Z;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TEx_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test2D_TEy ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//   CoordinateType ct2 = CoordinateType::Z;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 0;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TEy_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test2D_TEz ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//   CoordinateType ct2 = CoordinateType::Y;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 0;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TEz_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test2D_TMx ()
// {
//   CoordinateType ct1 = CoordinateType::Y;
//   CoordinateType ct2 = CoordinateType::Z;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 0;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TMx_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (!intScheme.getDoNeedHx ());
//   ASSERT (intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test2D_TMy ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//   CoordinateType ct2 = CoordinateType::Z;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 0;
//   FPValue angle3 = 0;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TMy_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (!intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (!intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test2D_TMz ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//   CoordinateType ct2 = CoordinateType::Y;
//
//   GridCoordinate2D overallSize (SIZE, SIZE, ct1, ct2);
//   GridCoordinate2D pmlSize (PML_SIZE, PML_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//   GridCoordinate2D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme2D_TMz_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (!intScheme.getDoNeedEx ());
//   ASSERT (!intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (intScheme.getDoNeedHy ());
//   ASSERT (!intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);
//
//   test2D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
// }
//
// template<LayoutType layout_type>
// void test3D ()
// {
//   CoordinateType ct1 = CoordinateType::X;
//   CoordinateType ct2 = CoordinateType::Y;
//   CoordinateType ct3 = CoordinateType::Z;
//
//   GridCoordinate3D overallSize (SIZE, SIZE, SIZE, ct1, ct2, ct3);
//   GridCoordinate3D pmlSize (PML_SIZE, PML_SIZE, PML_SIZE, ct1, ct2, ct3);
//   GridCoordinate3D tfsfSizeLeft (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);
//   GridCoordinate3D tfsfSizeRight (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);
//
//   bool useDoubleMaterialPrecision = false;
//
//   FPValue angle1 = 90;
//   FPValue angle2 = 90;
//   FPValue angle3 = 90;
//
//   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> yeeLayout (overallSize,
//                                                                         pmlSize,
//                                                                         tfsfSizeLeft,
//                                                                         tfsfSizeRight,
//                                                                         angle1 * PhysicsConst::Pi / 180.0,
//                                                                         angle2 * PhysicsConst::Pi / 180.0,
//                                                                         angle3 * PhysicsConst::Pi / 180.0,
//                                                                         useDoubleMaterialPrecision);
//
//   InternalScheme3D_3D_Grid<layout_type> intScheme;
//   intScheme.init (&yeeLayout, false);
//   intScheme.initScheme (DX, LAMBDA);
//
//   ASSERT (intScheme.getDoNeedEx ());
//   ASSERT (intScheme.getDoNeedEy ());
//   ASSERT (intScheme.getDoNeedEz ());
//   ASSERT (intScheme.getDoNeedHx ());
//   ASSERT (intScheme.getDoNeedHy ());
//   ASSERT (intScheme.getDoNeedHz ());
//
//   test<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid>
//     (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, ct3);
//
//   test3D<Grid> (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2, ct3);
// }

int main (int argc, char** argv)
{
  solverSettings.SetupFromCmd (argc, argv);

  test1D_ExHy<E_CENTERED> ();
  // test1D_ExHz<E_CENTERED> ();
  // test1D_EyHx<E_CENTERED> ();
  // test1D_EyHz<E_CENTERED> ();
  // test1D_EzHx<E_CENTERED> ();
  // test1D_EzHy<E_CENTERED> ();
  //
  // test2D_TEx<E_CENTERED> ();
  // test2D_TEy<E_CENTERED> ();
  // test2D_TEz<E_CENTERED> ();
  // test2D_TMx<E_CENTERED> ();
  // test2D_TMy<E_CENTERED> ();
  // test2D_TMz<E_CENTERED> ();
  //
  // test3D<E_CENTERED> ();

  solverSettings.Uninitialize ();

  return 0;
} /* main */

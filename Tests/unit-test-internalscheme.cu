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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void test (InternalScheme<Type, TCoord, layout_type> *cpuScheme,
           InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
{
  TCoord<grid_coord, true> zero (0, 0, 0
#ifdef DEBUG_INFO
                                 , cpuScheme->getType1 ()
                                 , cpuScheme->getType2 ()
                                 , cpuScheme->getType3 ()
#endif /* DEBUG_INFO */
                                 );
  TCoord<grid_coord, true> overallSize = cpuScheme->getEps ()->getSize ();

  for (time_step t = 0; t < SOLVER_SETTINGS.getNumTimeSteps (); ++t)
  {
    printf ("calculation %d time step\n", t);
    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);
      gpuScheme->performPlaneWaveEStepsKernelLaunch (t, zero1D, cpuScheme->getEInc ()->getSize ());
      gpuScheme->shiftInTimePlaneWaveKernelLaunchEInc (zero1D, cpuScheme->getEInc ()->getSize ());
      gpuScheme->nextTimeStepPlaneWaveKernelLaunchEInc ();
    }

    if (cpuScheme->getDoNeedEx ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::EX)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchEx (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchEx ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchDx (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchDx ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchD1x (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchD1x ();
      }
    }

    if (cpuScheme->getDoNeedEy ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::EY)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchEy (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchEy ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchDy (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchDy ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchD1y (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchD1y ();
      }
    }

    if (cpuScheme->getDoNeedEz ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::EZ)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchEz (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchEz ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchDz (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchDz ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchD1z (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchD1z ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);
      gpuScheme->performPlaneWaveHStepsKernelLaunch (t, zero1D, cpuScheme->getHInc ()->getSize ());
      gpuScheme->shiftInTimePlaneWaveKernelLaunchHInc (zero1D, cpuScheme->getHInc ()->getSize ());
      gpuScheme->nextTimeStepPlaneWaveKernelLaunchHInc ();
    }

    if (cpuScheme->getDoNeedHx ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::HX)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchHx (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchHx ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchBx (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchBx ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchB1x (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchB1x ();
      }
    }

    if (cpuScheme->getDoNeedHy ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::HY)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchHy (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchHy ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchBy (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchBy ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchB1y (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchB1y ();
      }
    }

    if (cpuScheme->getDoNeedHz ())
    {
      gpuScheme->template performFieldStepsKernelLaunch<static_cast<uint8_t> (GridType::HZ)> (t, zero, overallSize,
        cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());

      gpuScheme->shiftInTimeKernelLaunchHz (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
      gpuScheme->nextTimeStepKernelLaunchHz ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuScheme->shiftInTimeKernelLaunchBz (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchBz ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->shiftInTimeKernelLaunchB1z (zero, overallSize, cpuScheme->getType1 (), cpuScheme->getType2 (), cpuScheme->getType3 ());
        gpuScheme->nextTimeStepKernelLaunchB1z ();
      }
    }
  }
}

void test1D (Grid<GridCoordinate1D> *E,
             GridCoordinateFP1D diff,
             CoordinateType ct1)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      GridCoordinate1D pos = GRID_COORDINATE_1D(i, ct1);
      GridCoordinateFP1D posFP = GRID_COORDINATE_FP_1D (i, ct1);
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

void test2D (Grid<GridCoordinate2D> *E,
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
        GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, ct1, ct2);
        GridCoordinateFP2D posFP = GRID_COORDINATE_FP_2D (i, j, ct1, ct2);
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
void test3D (Grid<GridCoordinate3D> *E,
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
          GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k, ct1, ct2, ct3);
          GridCoordinateFP3D posFP = GRID_COORDINATE_FP_3D (i, j, k, ct1, ct2, ct3);
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

template<LayoutType layout_type>
void test1D_ExHy ()
{
  CoordinateType ct1 = CoordinateType::Z;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

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

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  intScheme.getEps ()->initialize (getFieldValueRealOnly (1.0));
  intScheme.getMu ()->initialize (getFieldValueRealOnly (1.0));

  GridCoordinate1D zero = GRID_COORDINATE_1D (0, ct1);
  GridCoordinate1D one = GRID_COORDINATE_1D (1, ct1);

  /*
   * Init InternalScheme on GPU
   */
  InternalSchemeGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> gpuScheme;
  gpuScheme.initFromCPU (&intScheme, overallSize, one);

  InternalSchemeGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> tmpGPUScheme;
  tmpGPUScheme.initOnGPU (&gpuScheme);

  InternalSchemeGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> *d_gpuScheme;
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuScheme, sizeof(InternalSchemeGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>)));

  /*
   * Copy InternalScheme to GPU
   */
  gpuScheme.copyFromCPU (zero, overallSize);

  tmpGPUScheme.copyToGPU (&gpuScheme);

  cudaCheckErrorCmd (cudaMemcpy (d_gpuScheme, &tmpGPUScheme, sizeof(InternalSchemeGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>), cudaMemcpyHostToDevice));

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED> (&intScheme, d_gpuScheme);

  /*
   * Copy back from GPU to CPU
   */
  gpuScheme.copyBackToCPU ();

  /*
   * Free memory
   */
  cudaCheckErrorCmd (cudaFree (d_gpuScheme));
  tmpGPUScheme.uninitOnGPU ();
  gpuScheme.uninitFromCPU ();

  test1D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1);
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
  SOLVER_SETTINGS.SetupFromCmd (argc, argv);

#ifdef DEBUG_INFO
  /*
   * This is required, because printf output is stored in a circular buffer of a fixed size.
   * If the buffer fills, old output will be overwritten. The buffer's size defaults to 1MB.
   * (see http://15418.courses.cs.cmu.edu/spring2013/article/15)
   */
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100*1024*1024);
#endif /* DEBUG_INFO */

  cudaCheckErrorCmd (SOLVER_SETTINGS.getNumCudaGPUs ());

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

  SOLVER_SETTINGS.Uninitialize ();

  return 0;
} /* main */

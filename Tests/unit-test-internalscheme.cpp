/*
 * Unit test for InternalScheme on CPU
 */

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
void test (InternalScheme<Type, TCoord, layout_type> *intScheme,
           TCoord<grid_coord, true> overallSize,
           TCoord<grid_coord, true> pmlSize,
           TCoord<grid_coord, true> tfsfSizeLeft,
           TCoord<grid_coord, true> tfsfSizeRight,
           CoordinateType ct1,
           CoordinateType ct2,
           CoordinateType ct3)
{
  intScheme->getEps ()->initialize (getFieldValueRealOnly (1.0));
  intScheme->getMu ()->initialize (getFieldValueRealOnly (1.0));

  TCoord<grid_coord, true> diff (1, 1, 1
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                 );

  TCoord<grid_coord, true> zero (0, 0, 0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                 );

  for (time_step t = 0; t < SOLVER_SETTINGS.getNumTimeSteps (); ++t)
  {
    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D (0
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif /* DEBUG_INFO */
                               );

      intScheme->performPlaneWaveESteps (t, zero1D, intScheme->getEInc ()->getSize ());
      intScheme->getEInc ()->shiftInTime (zero1D, intScheme->getEInc ()->getSize ());
      intScheme->getEInc ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEx ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, diff, overallSize - diff);

      intScheme->getEx ()->shiftInTime (zero, intScheme->getEx ()->getSize ());
      intScheme->getEx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDx ()->shiftInTime (zero, intScheme->getDx ()->getSize ());
        intScheme->getDx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1x ()->shiftInTime (zero, intScheme->getD1x ()->getSize ());
        intScheme->getD1x ()->nextTimeStep ();
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, diff, overallSize - diff);

      intScheme->getEy ()->shiftInTime (zero, intScheme->getEy ()->getSize ());
      intScheme->getEy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDy ()->shiftInTime (zero, intScheme->getDy ()->getSize ());
        intScheme->getDy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1y ()->shiftInTime (zero, intScheme->getD1y ()->getSize ());
        intScheme->getD1y ()->nextTimeStep ();
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, diff, overallSize - diff);

      intScheme->getEz ()->shiftInTime (zero, intScheme->getEz ()->getSize ());
      intScheme->getEz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDz ()->shiftInTime (zero, intScheme->getDz ()->getSize ());
        intScheme->getDz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1z ()->shiftInTime (zero, intScheme->getD1z ()->getSize ());
        intScheme->getD1z ()->nextTimeStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D (0
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif /* DEBUG_INFO */
                               );

      intScheme->performPlaneWaveHSteps (t, zero1D, intScheme->getHInc ()->getSize ());
      intScheme->getHInc ()->shiftInTime (zero1D, intScheme->getHInc ()->getSize ());
      intScheme->getHInc ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHx ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, diff, overallSize - diff);

      intScheme->getHx ()->shiftInTime (zero, intScheme->getHx ()->getSize ());
      intScheme->getHx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBx ()->shiftInTime (zero, intScheme->getBx ()->getSize ());
        intScheme->getBx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1x ()->shiftInTime (zero, intScheme->getB1x ()->getSize ());
        intScheme->getB1x ()->nextTimeStep ();
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, diff, overallSize - diff);

      intScheme->getHy ()->shiftInTime (zero, intScheme->getHy ()->getSize ());
      intScheme->getHy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBy ()->shiftInTime (zero, intScheme->getBy ()->getSize ());
        intScheme->getBy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1y ()->shiftInTime (zero, intScheme->getB1y ()->getSize ());
        intScheme->getB1y ()->nextTimeStep ();
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      intScheme->template performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, diff, overallSize - diff);

      intScheme->getHz ()->shiftInTime (zero, intScheme->getHz ()->getSize ());
      intScheme->getHz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBz ()->shiftInTime (zero, intScheme->getBz ()->getSize ());
        intScheme->getBz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1z ()->shiftInTime (zero, intScheme->getB1z ()->getSize ());
        intScheme->getB1z ()->nextTimeStep ();
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
      GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                            , ct1
#endif /* DEBUG_INFO */
                            );
      GridCoordinateFP1D posFP (i
#ifdef DEBUG_INFO
                                , ct1
#endif /* DEBUG_INFO */
                                );
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
        GridCoordinate2D pos (i, j
#ifdef DEBUG_INFO
                              , ct1, ct2
#endif /* DEBUG_INFO */
                              );
        GridCoordinateFP2D posFP (i, j
#ifdef DEBUG_INFO
                                  , ct1, ct2
#endif /* DEBUG_INFO */
                                  );
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
          GridCoordinate3D pos (i, j, k
#ifdef DEBUG_INFO
                                , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                );
          GridCoordinateFP3D posFP (i, j, k
#ifdef DEBUG_INFO
                                    , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                    );
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

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1);
}

template<LayoutType layout_type>
void test1D_ExHz ()
{
  CoordinateType ct1 = CoordinateType::Y;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1);
}

template<LayoutType layout_type>
void test1D_EyHx ()
{
  CoordinateType ct1 = CoordinateType::Z;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 0;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1);
}

template<LayoutType layout_type>
void test1D_EyHz ()
{
  CoordinateType ct1 = CoordinateType::X;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1);
}

template<LayoutType layout_type>
void test1D_EzHx ()
{
  CoordinateType ct1 = CoordinateType::Y;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1);
}

template<LayoutType layout_type>
void test1D_EzHy ()
{
  CoordinateType ct1 = CoordinateType::X;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  test1D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1);
}

template<LayoutType layout_type>
void test2D_TEx ()
{
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test2D_TEy ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test2D_TEz ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test2D_TMx ()
{
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test2D_TMy ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test2D_TMz ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}

template<LayoutType layout_type>
void test3D ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;
  CoordinateType ct3 = CoordinateType::Z;

  GridCoordinate3D overallSize = GRID_COORDINATE_3D (SIZE, SIZE, SIZE, ct1, ct2, ct3);
  GridCoordinate3D pmlSize = GRID_COORDINATE_3D (PML_SIZE, PML_SIZE, PML_SIZE, ct1, ct2, ct3);
  GridCoordinate3D tfsfSizeLeft = GRID_COORDINATE_3D (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);
  GridCoordinate3D tfsfSizeRight = GRID_COORDINATE_3D (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, ct3);

  test3D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2, ct3);
}

int main (int argc, char** argv)
{
  solverSettings.SetupFromCmd (argc, argv);

  test1D_ExHy<E_CENTERED> ();
  test1D_ExHz<E_CENTERED> ();
  test1D_EyHx<E_CENTERED> ();
  test1D_EyHz<E_CENTERED> ();
  test1D_EzHx<E_CENTERED> ();
  test1D_EzHy<E_CENTERED> ();

  test2D_TEx<E_CENTERED> ();
  test2D_TEy<E_CENTERED> ();
  test2D_TEz<E_CENTERED> ();
  test2D_TMx<E_CENTERED> ();
  test2D_TMy<E_CENTERED> ();
  test2D_TMz<E_CENTERED> ();

  test3D<E_CENTERED> ();

  solverSettings.Uninitialize ();

  return 0;
} /* main */

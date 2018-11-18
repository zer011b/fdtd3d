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

  if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
  {
    if (intScheme->getDoNeedEx ())
    {
      for (grid_coord i = 0; i < intScheme->getEx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getEx ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getExStartDiff () && pos < intScheme->getEx ()->getSize () - intScheme->getYeeLayout ()->getExEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEx ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EX, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEx ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getCbEx ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      for (grid_coord i = 0; i < intScheme->getEy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getEy ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getEyStartDiff () && pos < intScheme->getEy ()->getSize () - intScheme->getYeeLayout ()->getEyEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEy ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EY, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEy ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getCbEy ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      for (grid_coord i = 0; i < intScheme->getEz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getEz ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getEzStartDiff () && pos < intScheme->getEz ()->getSize () - intScheme->getYeeLayout ()->getEzEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEz ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EZ, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEz ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getCbEz ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      for (grid_coord i = 0; i < intScheme->getHx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getHx ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getHxStartDiff () && pos < intScheme->getHx ()->getSize () - intScheme->getYeeLayout ()->getHxEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHx ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HX, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHx ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getDbHx ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      for (grid_coord i = 0; i < intScheme->getHy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getHy ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getHyStartDiff () && pos < intScheme->getHy ()->getSize () - intScheme->getYeeLayout ()->getHyEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHy ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HY, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHy ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getDbHy ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      for (grid_coord i = 0; i < intScheme->getHz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TCoord<grid_coord, true> pos = intScheme->getHz ()->calculatePositionFromIndex (i);

        if (!(pos >= intScheme->getYeeLayout ()->getHzStartDiff () && pos < intScheme->getHz ()->getSize () - intScheme->getYeeLayout ()->getHzEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHz ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HZ, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHz ()->setFieldValue (FIELDVALUE (ca, 0), i, 0);
        intScheme->getDbHz ()->setFieldValue (FIELDVALUE (cb, 0), i, 0);
      }
    }
  }

  for (time_step t = 0; t < SOLVER_SETTINGS.getNumTimeSteps (); ++t)
  {
    DPRINTF (LOG_LEVEL_NONE, "calculating time step %d\n", t);

    TCoord<grid_coord, true> ExStart = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationStart (intScheme->getYeeLayout ()->getExStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> ExEnd = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationEnd (intScheme->getYeeLayout ()->getExEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    TCoord<grid_coord, true> EyStart = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationStart (intScheme->getYeeLayout ()->getEyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> EyEnd = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationEnd (intScheme->getYeeLayout ()->getEyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    TCoord<grid_coord, true> EzStart = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationStart (intScheme->getYeeLayout ()->getEzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> EzEnd = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationEnd (intScheme->getYeeLayout ()->getEzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    TCoord<grid_coord, true> HxStart = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationStart (intScheme->getYeeLayout ()->getHxStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> HxEnd = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationEnd (intScheme->getYeeLayout ()->getHxEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    TCoord<grid_coord, true> HyStart = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationStart (intScheme->getYeeLayout ()->getHyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> HyEnd = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationEnd (intScheme->getYeeLayout ()->getHyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    TCoord<grid_coord, true> HzStart = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationStart (intScheme->getYeeLayout ()->getHzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    TCoord<grid_coord, true> HzEnd = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationEnd (intScheme->getYeeLayout ()->getHzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

      intScheme->performPlaneWaveESteps (t, zero1D, intScheme->getEInc ()->getSize ());
      intScheme->getEInc ()->shiftInTime ();
      intScheme->getEInc ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEx ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::EX> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (ExStart, ExEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getEx ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::EX> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEx (), coordFP,
                                                               intScheme->getHz (), intScheme->getHy (), NULLPTR,
                                                               intScheme->getCaEx (), intScheme->getCbEx (),
                                                               false,
                                                               GridType::EX, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }
      }

      intScheme->getEx ()->shiftInTime ();
      intScheme->getEx ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEy ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::EY> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (EyStart, EyEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getEy ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::EY> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEy (), coordFP,
                                                               intScheme->getHx (), intScheme->getHz (), NULLPTR,
                                                               intScheme->getCaEy (), intScheme->getCbEy (),
                                                               false,
                                                               GridType::EY, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }
      }

      intScheme->getEy ()->shiftInTime ();
      intScheme->getEy ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedEz ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::EZ> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (EzStart, EzEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getEz ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::EZ> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEz (), coordFP,
                                                               intScheme->getHy (), intScheme->getHx (), NULLPTR,
                                                               intScheme->getCaEz (), intScheme->getCbEz (),
                                                               false,
                                                               GridType::EZ, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }
      }

      intScheme->getEz ()->shiftInTime ();
      intScheme->getEz ()->nextTimeStep ();
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

      intScheme->performPlaneWaveHSteps (t, zero1D, intScheme->getHInc ()->getSize ());
      intScheme->getHInc ()->shiftInTime ();
      intScheme->getHInc ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHx ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::HX> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HxStart, HxEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getHx ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::HX> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHx (), coordFP,
                                                               intScheme->getEy (), intScheme->getEz (), NULLPTR,
                                                               intScheme->getDaHx (), intScheme->getDbHx (),
                                                               false,
                                                               GridType::HX, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }
      }

      intScheme->getHx ()->shiftInTime ();
      intScheme->getHx ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHy ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::HY> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HyStart, HyEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getHy ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::HY> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHy (), coordFP,
                                                               intScheme->getEz (), intScheme->getEx (), NULLPTR,
                                                               intScheme->getDaHy (), intScheme->getDbHy (),
                                                               false,
                                                               GridType::HY, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }
      }

      intScheme->getHy ()->shiftInTime ();
      intScheme->getHy ()->nextTimeStep ();
    }

    if (intScheme->getDoNeedHz ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->calculateFieldStepInitDiff<GridType::HZ> (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HzStart, HzEnd, start3D, end3D, ct1, ct2, ct3);

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            TCoord<grid_coord, true> posAbs = intScheme->getHz ()->getTotalPosition (pos);
            intScheme->calculateFieldStepIteration<GridType::HZ> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHz (), coordFP,
                                                               intScheme->getEx (), intScheme->getEy (), NULLPTR,
                                                               intScheme->getDaHz (), intScheme->getDbHz (),
                                                               false,
                                                               GridType::HZ, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }
      }

      intScheme->getHz ()->shiftInTime ();
      intScheme->getHz ()->nextTimeStep ();
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

      FieldValue val = *E->getFieldValue (pos, 1);

      if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE)
      {
        ASSERT (SQR (val.abs () - FPValue (1)) < 0.0001);
      }
      else
      {
        ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
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

        FieldValue val = *E->getFieldValue (pos, 1);

        if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
            && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE)
        {
          ASSERT (SQR (val.abs () - FPValue (1)) < 0.0001);
        }
        else
        {
          ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
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

          FieldValue val = *E->getFieldValue (pos, 1);

          if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
              && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE
              && posFP.get3 () >= TFSF_SIZE && posFP.get3 () <= SIZE - TFSF_SIZE)
          {
            ASSERT (SQR (val.abs () - FPValue (1)) < 0.0001);
          }
          else
          {
            ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
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

  /*
   * PML mode is not supported (Sigmas, Ca, Cb are not initialized here)
   */
  ASSERT (!solverSettings.getDoUsePML ());

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

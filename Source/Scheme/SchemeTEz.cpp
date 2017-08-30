#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "Kernels.h"
#include "SchemeTEz.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_2D

void
SchemeTEz::performPlaneWaveESteps (time_step t)
{
  for (grid_coord i = 1; i < EInc.getSize ().getX (); ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valE = EInc.getFieldPointValue (pos);

    GridCoordinate1D posLeft (i - 1);
    GridCoordinate1D posRight (i);

    FieldPointValue *valH1 = HInc.getFieldPointValue (posLeft);
    FieldPointValue *valH2 = HInc.getFieldPointValue (posRight);

    FPValue S = 1 / courantNum;
    FPValue stepWaveLength = PhysicsConst::SpeedOfLight / (sourceFrequency * gridStep);
    FPValue arg = PhysicsConst::Pi * S / stepWaveLength;

    FPValue relPhi;
    if (incidentWaveAngle == PhysicsConst::Pi / 4)
    {
      relPhi = sqrt (2) * asin (sin(arg) / (S * sqrt (2))) / asin (sin (arg) / S);
    }
    else
    {
      ASSERT (incidentWaveAngle == 0);

      relPhi = 1;
    }

    FieldValue val = valE->getPrevValue () + (gridTimeStep / (relPhi * PhysicsConst::Eps0 * gridStep)) * (valH1->getPrevValue () - valH2->getPrevValue ());

    valE->setCurValue (val);
  }

  EInc.nextTimeStep ();
}

void
SchemeTEz::performPlaneWaveHSteps (time_step t)
{
  for (grid_coord i = 0; i < HInc.getSize ().getX () - 1; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valH = HInc.getFieldPointValue (pos);

    GridCoordinate1D posLeft (i);
    GridCoordinate1D posRight (i + 1);

    FieldPointValue *valE1 = EInc.getFieldPointValue (posLeft);
    FieldPointValue *valE2 = EInc.getFieldPointValue (posRight);

    FPValue S = 1 / courantNum;
    FPValue stepWaveLength = PhysicsConst::SpeedOfLight / (sourceFrequency * gridStep);
    FPValue arg = PhysicsConst::Pi * S / stepWaveLength;

    FPValue relPhi;
    if (incidentWaveAngle == PhysicsConst::Pi / 4)
    {
      relPhi = sqrt (2) * asin (sin(arg) / (S * sqrt (2))) / asin (sin (arg) / S);
    }
    else
    {
      ASSERT (incidentWaveAngle == 0);

      relPhi = 1;
    }

    FieldValue val = valH->getPrevValue () + (gridTimeStep / (relPhi * PhysicsConst::Mu0 * gridStep)) * (valE1->getPrevValue () - valE2->getPrevValue ());

    valH->setCurValue (val);
  }

  GridCoordinate1D pos (0);
  FieldPointValue *valH = HInc.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
  valH->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                 cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
  valH->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */

  HInc.nextTimeStep ();
}

void
SchemeTEz::performExSteps (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateExStepPML (t, ExStart, ExEnd);
  }
  else
  {
    calculateExStep (t, ExStart, ExEnd);
  }
}

void
SchemeTEz::calculateExStep (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ex.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getExCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valEx = Ex.getFieldPointValue (pos);

      GridCoordinate2D posDown = yeeLayout->getExCircuitElement (expand (pos), LayoutDirection::DOWN).shrink ();
      GridCoordinate2D posUp = yeeLayout->getExCircuitElement (expand (pos), LayoutDirection::UP).shrink ();

      FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_down = yeeLayout->doNeedTFSFUpdateExBorder (expand (posAbs), LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout->doNeedTFSFUpdateExBorder (expand (posAbs), LayoutDirection::UP, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_down)
        {
          auxPos = posUp;
        }
        else if (do_need_update_up)
        {
          auxPos = posDown;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (Hz.getTotalPosition (auxPos))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevHz1 -= -diff;
        }
        else if (do_need_update_up)
        {
          prevHz2 -= -diff;
        }
      }

      FieldValue val = calculateEx_2D_TEz (valEx->getPrevValue (),
                                           prevHz1,
                                           prevHz2,
                                           gridTimeStep,
                                           gridStep,
                                           eps * eps0);

      valEx->setCurValue (val);
    }
  }
}

void
SchemeTEz::calculateExStepPML (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ex.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getExCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valDx = Dx.getFieldPointValue (pos);

      GridCoordinate2D posDown = yeeLayout->getExCircuitElement (expand (pos), LayoutDirection::DOWN).shrink ();
      GridCoordinate2D posUp = yeeLayout->getExCircuitElement (expand (pos), LayoutDirection::UP).shrink ();

      FieldPointValue* valSigmaY1 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaY2 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaY = (valSigmaY1->getCurValue ().real () + valSigmaY2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaY = (valSigmaY1->getCurValue () + valSigmaY2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_down = yeeLayout->doNeedTFSFUpdateExBorder (expand (posAbs), LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout->doNeedTFSFUpdateExBorder (expand (posAbs), LayoutDirection::UP, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_down)
        {
          auxPos = posUp;
        }
        else if (do_need_update_up)
        {
          auxPos = posDown;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (Hz.getTotalPosition (auxPos))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevHz1 -= -diff;
        }
        else if (do_need_update_up)
        {
          prevHz2 -= -diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_y = 1;

      FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

      FieldValue val = calculateEx_2D_TEz_Precalc (valDx->getPrevValue (),
                                                   prevHz1,
                                                   prevHz2,
                                                   Ca,
                                                   Cb);

      valDx->setCurValue (val);
    }
  }

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ex.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getExCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valEx = Ex.getFieldPointValue (pos);
      FieldPointValue* valDx = Dx.getFieldPointValue (pos);

      FieldPointValue* valSigmaX1 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaX2 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valSigmaZ1 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaZ2 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue ().real () + valSigmaX2->getCurValue ().real ()) / 2;
      FPValue sigmaZ = (valSigmaZ1->getCurValue ().real () + valSigmaZ2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue () + valSigmaX2->getCurValue ()) / 2;
      FPValue sigmaZ = (valSigmaZ1->getCurValue () + valSigmaZ2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_x = 1;
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_x + sigmaX * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_x - sigmaX * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

      FieldValue val = calculateEx_from_Dx_Precalc (valEx->getPrevValue (),
                                                    valDx->getCurValue (),
                                                    valDx->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valEx->setCurValue (val);
    }
  }
}

void
SchemeTEz::performEySteps (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateEyStepPML (t, EyStart, EyEnd);
  }
  else
  {
    calculateEyStep (t, EyStart, EyEnd);
  }
}

void
SchemeTEz::calculateEyStep (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ey.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getEyCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valEy = Ey.getFieldPointValue (pos);

      GridCoordinate2D posLeft = yeeLayout->getEyCircuitElement (expand (pos), LayoutDirection::LEFT).shrink ();
      GridCoordinate2D posRight = yeeLayout->getEyCircuitElement (expand (pos), LayoutDirection::RIGHT).shrink ();

      FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout->doNeedTFSFUpdateEyBorder (expand (posAbs), LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout->doNeedTFSFUpdateEyBorder (expand (posAbs), LayoutDirection::RIGHT, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_left)
        {
          auxPos = posRight;
        }
        else if (do_need_update_right)
        {
          auxPos = posLeft;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (Hz.getTotalPosition (auxPos))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevHz1 -= -diff;
        }
        else if (do_need_update_right)
        {
          prevHz2 -= -diff;
        }
      }

      FieldValue val = calculateEy_2D_TEz (valEy->getPrevValue (),
                                           prevHz1,
                                           prevHz2,
                                           gridTimeStep,
                                           gridStep,
                                           eps * eps0);

      valEy->setCurValue (val);
    }
  }
}

void
SchemeTEz::calculateEyStepPML (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ey.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getEyCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valDy = Dy.getFieldPointValue (pos);

      GridCoordinate2D posLeft = yeeLayout->getEyCircuitElement (expand (pos), LayoutDirection::LEFT).shrink ();
      GridCoordinate2D posRight = yeeLayout->getEyCircuitElement (expand (pos), LayoutDirection::RIGHT).shrink ();

      FieldPointValue* valSigmaZ1 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaZ2 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaZ = (valSigmaZ1->getCurValue ().real () + valSigmaZ2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaZ = (valSigmaZ1->getCurValue () + valSigmaZ2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout->doNeedTFSFUpdateEyBorder (expand (posAbs), LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout->doNeedTFSFUpdateEyBorder (expand (posAbs), LayoutDirection::RIGHT, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_left)
        {
          auxPos = posRight;
        }
        else if (do_need_update_right)
        {
          auxPos = posLeft;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (Hz.getTotalPosition (auxPos))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevHz1 -= -diff;
        }
        else if (do_need_update_right)
        {
          prevHz2 -= -diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

      FieldValue val = calculateEy_2D_TEz_Precalc (valDy->getPrevValue (),
                                                   prevHz1,
                                                   prevHz2,
                                                   Ca,
                                                   Cb);

      valDy->setCurValue (val);
    }
  }

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ey.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getEyCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valEy = Ey.getFieldPointValue (pos);

      FieldPointValue* valDy = Dy.getFieldPointValue (pos);

      FieldPointValue* valSigmaX1 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaX2 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valSigmaY1 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaY2 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout->getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout->getMinEpsCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue ().real () + valSigmaX2->getCurValue ().real ()) / 2;
      FPValue sigmaY = (valSigmaY1->getCurValue ().real () + valSigmaY2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue () + valSigmaX2->getCurValue ()) / 2;
      FPValue sigmaY = (valSigmaY1->getCurValue () + valSigmaY2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_x = 1;
      FPValue k_y = 1;

      FPValue Ca = (2 * eps0 * k_x - sigmaX * gridTimeStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_y + sigmaY * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_y - sigmaY * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);

      FieldValue val = calculateEy_from_Dy_Precalc (valEy->getPrevValue (),
                                                    valDy->getCurValue (),
                                                    valDy->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valEy->setCurValue (val);
    }
  }
}

void
SchemeTEz::performHzSteps (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateHzStepPML (t, HzStart, HzEnd);
  }
  else
  {
    calculateHzStep (t, HzStart, HzEnd);
  }
}

void
SchemeTEz::calculateHzStep (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hz.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);

      GridCoordinate2D posLeft = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::LEFT).shrink ();
      GridCoordinate2D posRight = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::RIGHT).shrink ();

      GridCoordinate2D posDown = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::DOWN).shrink ();
      GridCoordinate2D posUp = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::UP).shrink ();

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu3 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu4 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real () + valMu3->getCurValue ().real () + valMu4->getCurValue ().real ()) / 4;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEy1 = Ey.getFieldPointValue (posRight);
      FieldPointValue* valEy2 = Ey.getFieldPointValue (posLeft);

      FieldPointValue* valEx1 = Ex.getFieldPointValue (posUp);
      FieldPointValue* valEx2 = Ex.getFieldPointValue (posDown);

      FieldValue prevEx1 = valEx1->getPrevValue ();
      FieldValue prevEx2 = valEx2->getPrevValue ();

      FieldValue prevEy1 = valEy1->getPrevValue ();
      FieldValue prevEy2 = valEy2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::RIGHT, false);
        bool do_need_update_down = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::UP, false);

        GridCoordinate2D auxPosX;
        GridCoordinate2D auxPosY;
        FieldValue diffX;
        FieldValue diffY;

        if (do_need_update_left)
        {
          auxPosX = posLeft;
        }
        else if (do_need_update_right)
        {
          auxPosX = posRight;
        }

        if (do_need_update_down)
        {
          auxPosY = posDown;
        }
        else if (do_need_update_up)
        {
          auxPosY = posUp;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getExCoordFP (expand (Ex.getTotalPosition (auxPosY))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          diffY = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getEyCoordFP (expand (Ey.getTotalPosition (auxPosX))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          diffX = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevEx2 += diffY * sin (incidentWaveAngle);
        }
        else if (do_need_update_up)
        {
          prevEx1 += diffY * sin (incidentWaveAngle);
        }

        if (do_need_update_left)
        {
          prevEy2 += -diffX * cos (incidentWaveAngle);
        }
        else if (do_need_update_right)
        {
          prevEy1 += -diffX * cos (incidentWaveAngle);
        }
      }

      FieldValue val = calculateHz_3D (valHz->getPrevValue (),
                                       prevEx1,
                                       prevEx2,
                                       prevEy1,
                                       prevEy2,
                                       gridTimeStep,
                                       gridStep,
                                       mu * mu0);

      valHz->setCurValue (val);
    }
  }
}

void
SchemeTEz::calculateHzStepPML (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hz.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valBz = Bz.getFieldPointValue (pos);

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);

      GridCoordinate2D posLeft = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::LEFT).shrink ();
      GridCoordinate2D posRight = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::RIGHT).shrink ();

      GridCoordinate2D posDown = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::DOWN).shrink ();
      GridCoordinate2D posUp = yeeLayout->getHzCircuitElement (expand (pos), LayoutDirection::UP).shrink ();

      FieldPointValue* valSigmaX1 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaX2 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaX3 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaX4 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaX = (valSigmaX1->getCurValue ().real () + valSigmaX2->getCurValue ().real () + valSigmaX3->getCurValue ().real () + valSigmaX4->getCurValue ().real ()) / 4;
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaX = (valSigmaX1->getCurValue () + valSigmaX2->getCurValue () + valSigmaX3->getCurValue () + valSigmaX4->getCurValue ()) / 4;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEy1 = Ey.getFieldPointValue (posRight);
      FieldPointValue* valEy2 = Ey.getFieldPointValue (posLeft);

      FieldPointValue* valEx1 = Ex.getFieldPointValue (posUp);
      FieldPointValue* valEx2 = Ex.getFieldPointValue (posDown);

      FieldValue prevEx1 = valEx1->getPrevValue ();
      FieldValue prevEx2 = valEx2->getPrevValue ();

      FieldValue prevEy1 = valEy1->getPrevValue ();
      FieldValue prevEy2 = valEy2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::RIGHT, false);
        bool do_need_update_down = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout->doNeedTFSFUpdateHzBorder (expand (posAbs), LayoutDirection::UP, false);

        GridCoordinate2D auxPosX;
        GridCoordinate2D auxPosY;
        FieldValue diffX;
        FieldValue diffY;

        if (do_need_update_left)
        {
          auxPosX = posLeft;
        }
        else if (do_need_update_right)
        {
          auxPosX = posRight;
        }

        if (do_need_update_down)
        {
          auxPosY = posDown;
        }
        else if (do_need_update_up)
        {
          auxPosY = posUp;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getExCoordFP (expand (Ex.getTotalPosition (auxPosY))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          diffY = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = yeeLayout->getEyCoordFP (expand (Ey.getTotalPosition (auxPosX))).shrink ();
          GridCoordinateFP2D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

          FPValue x = realCoord.getX () - zeroCoordFP.getX ();
          FPValue y = realCoord.getY () - zeroCoordFP.getY ();
          FPValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FPValue coordD1 = (FPValue) ((grid_iter) d);
          FPValue coordD2 = coordD1 + 1;
          FPValue proportionD2 = d - coordD1;
          FPValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          diffX = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevEx2 += diffY * sin (incidentWaveAngle);
        }
        else if (do_need_update_up)
        {
          prevEx1 += diffY * sin (incidentWaveAngle);
        }

        if (do_need_update_left)
        {
          prevEy2 += -diffX * cos (incidentWaveAngle);
        }
        else if (do_need_update_right)
        {
          prevEy1 += -diffX * cos (incidentWaveAngle);
        }
      }

      /*
       * FIXME: precalculate coefficients
       */

      FPValue k_x = 1;

      FPValue Ca = (2 * eps0 * k_x - sigmaX * gridTimeStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);

      FieldValue val = calculateHz_3D_Precalc (valBz->getPrevValue (),
                                               prevEx1,
                                               prevEx2,
                                               prevEy1,
                                               prevEy2,
                                               Ca,
                                               Cb);

      valBz->setCurValue (val);
    }
  }

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hz.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = yeeLayout->getHzCoordFP (expand (posAbs)).shrink ();

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);

      FieldPointValue* valBz = Bz.getFieldPointValue (pos);

      FieldPointValue* valSigmaY1 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaY2 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaY3 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaY4 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valSigmaZ1 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaZ2 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaZ3 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valSigmaZ4 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu3 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));
      FieldPointValue* valMu4 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout->getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout->getMinMuCoordFP ().getZ ())).shrink ()));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real () + valMu3->getCurValue ().real () + valMu4->getCurValue ().real ()) / 4;

      FPValue sigmaY = (valSigmaY1->getCurValue ().real () + valSigmaY2->getCurValue ().real () + valSigmaY3->getCurValue ().real () + valSigmaY4->getCurValue ().real ()) / 4;
      FPValue sigmaZ = (valSigmaZ1->getCurValue ().real () + valSigmaZ2->getCurValue ().real () + valSigmaZ3->getCurValue ().real () + valSigmaZ4->getCurValue ().real ()) / 4;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;

      FPValue sigmaY = (valSigmaY1->getCurValue () + valSigmaY2->getCurValue () + valSigmaY3->getCurValue () + valSigmaY4->getCurValue ()) / 4;
      FPValue sigmaZ = (valSigmaZ1->getCurValue () + valSigmaZ2->getCurValue () + valSigmaZ3->getCurValue () + valSigmaZ4->getCurValue ()) / 4;
#endif /* !COMPLEX_FIELD_VALUES */

      /*
       * FIXME: precalculate coefficients
       */

      FPValue k_y = 1;
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_z + sigmaZ * gridTimeStep) / (mu * mu0)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_z - sigmaZ * gridTimeStep) / (mu * mu0)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

      FieldValue val = calculateHz_from_Bz_Precalc (valHz->getPrevValue (),
                                                    valBz->getCurValue (),
                                                    valBz->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valHz->setCurValue (val);
    }
  }
}

void
SchemeTEz::performNSteps (time_step startStep, time_step numberTimeSteps)
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  GridCoordinate2D HzSize = Hz.getSize ();

  time_step stepLimit = startStep + numberTimeSteps;

  for (int t = startStep; t < stepLimit; ++t)
  {
    GridCoordinate3D ExStart = expand (Ex.getComputationStart (yeeLayout->getExStartDiff ()));
    GridCoordinate3D ExEnd = expand (Ex.getComputationEnd (yeeLayout->getExEndDiff ()));

    GridCoordinate3D EyStart = expand (Ey.getComputationStart (yeeLayout->getEyStartDiff ()));
    GridCoordinate3D EyEnd = expand (Ey.getComputationEnd (yeeLayout->getEyEndDiff ()));

    GridCoordinate3D HzStart = expand (Hz.getComputationStart (yeeLayout->getHzStartDiff ()));
    GridCoordinate3D HzEnd = expand (Hz.getComputationEnd (yeeLayout->getHzEndDiff ()));

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();

    if (usePML)
    {
      Dx.nextTimeStep ();
      Dy.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHzSteps (t, HzStart, HzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      if (processId == 0)
#endif
      {
        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
      }
    }

    Hz.nextTimeStep ();

    if (usePML)
    {
      Bz.nextTimeStep ();
    }

    if (t % 500 == 0)
    {
      if (dumpRes)
      {
        BMPDumper<GridCoordinate2D> dumperEx;
        dumperEx.init (t, CURRENT, processId, "2D-TEz-in-time-Ex");
        dumperEx.dumpGrid (Ex, GridCoordinate2D (0, 0), Ex.getSize ());

        BMPDumper<GridCoordinate2D> dumperEy;
        dumperEy.init (t, CURRENT, processId, "2D-TEz-in-time-Ey");
        dumperEy.dumpGrid (Ey, GridCoordinate2D (0, 0), Ey.getSize ());

        BMPDumper<GridCoordinate2D> dumperHz;
        dumperHz.init (t, CURRENT, processId, "2D-TEz-in-time-Hz");
        dumperHz.dumpGrid (Hz, GridCoordinate2D (0, 0), Hz.getSize ());
      }
    }

#ifdef COMPLEX_FIELD_VALUES
    if (t % 100 == 0)
    {
      if (dumpRes)
      {
        FPValue norm = 0;

        for (grid_iter i = 0; i < Hz.getSize ().getX (); ++i)
        {
          for (grid_iter j = 0; j < Hz.getSize ().getY (); ++j)
          {
            GridCoordinate2D pos (i, j);

            if (!yeeLayout->isHzInPML (expand (Hz.getTotalPosition (pos))))
            {
              FieldPointValue *value = Hz.getFieldPointValue (pos);

              FieldValue val = value->getCurValue ();

              norm += val.real () * val.real () + val.imag () * val.imag ();
            }
          }
        }

        norm = sqrt (norm);

        printf ("Norm at step %u: " PRINTF_MODIFIER "\n", t, norm);
      }
    }
#endif /* COMPLEX_FIELD_VALUES */
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumperEx;
    dumperEx.init (stepLimit, CURRENT, processId, "2D-TEz-in-time-Ex");
    dumperEx.dumpGrid (Ex, GridCoordinate2D (0, 0), Ex.getSize ());

    BMPDumper<GridCoordinate2D> dumperEy;
    dumperEy.init (stepLimit, CURRENT, processId, "2D-TEz-in-time-Ey");
    dumperEy.dumpGrid (Ey, GridCoordinate2D (0, 0), Ey.getSize ());

    BMPDumper<GridCoordinate2D> dumperHz;
    dumperHz.init (stepLimit, CURRENT, processId, "2D-TEz-in-time-Hz");
    dumperHz.dumpGrid (Hz, GridCoordinate2D (0, 0), Hz.getSize ());
  }

#ifdef COMPLEX_FIELD_VALUES
  if (dumpRes)
  {
    FPValue norm = 0;

    for (grid_iter i = 0; i < Hz.getSize ().getX (); ++i)
    {
      for (grid_iter j = 0; j < Hz.getSize ().getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout->isHzInPML (expand (Hz.getTotalPosition (pos))))
        {
          FieldPointValue *value = Hz.getFieldPointValue (pos);

          FieldValue val = value->getCurValue ();

          norm += val.real () * val.real () + val.imag () * val.imag ();
        }
      }
    }

    norm = sqrt (norm);

    printf ("Norm at step %u: " PRINTF_MODIFIER "\n", stepLimit, norm);
  }
#endif /* COMPLEX_FIELD_VALUES */
}

void
SchemeTEz::performAmplitudeSteps (time_step startStep)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  int is_stable_state = 0;

  GridCoordinate2D HzSize = Hz.getSize ();

  time_step t = startStep;

  GridCoordinate2D PMLSize = yeeLayout->getLeftBorderPML ().shrink ();

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FPValue maxAccuracy = -1;

    //is_stable_state = 1;

    GridCoordinate3D ExStart = Ex.getComputationStart (yeeLayout->getExStartDiff ());
    GridCoordinate3D ExEnd = Ex.getComputationEnd (yeeLayout->getExEndDiff ());

    GridCoordinate3D EyStart = Ey.getComputationStart (yeeLayout->getEyStartDiff ());
    GridCoordinate3D EyEnd = Ey.getComputationEnd (yeeLayout->getEyEndDiff ());

    GridCoordinate3D HzStart = Hz.getComputationStart (yeeLayout->getHzStartDiff ());
    GridCoordinate3D HzEnd = Hz.getComputationEnd (yeeLayout->getHzEndDiff ());

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);

    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout->isExInPML (Ex.getTotalPosition (pos)))
        {
          FieldPointValue* tmp = Ex.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = ExAmplitude.getFieldPointValue (pos);

          FPValue val = tmp->getCurValue ();

          if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
    {
      for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout->isEyInPML (Ey.getTotalPosition (pos)))
        {
          FieldPointValue* tmp = Ey.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = EyAmplitude.getFieldPointValue (pos);

          FPValue val = tmp->getCurValue ();

          if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();

    if (usePML)
    {
      Dx.nextTimeStep ();
      Dy.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHzSteps (t, HzStart, HzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      if (processId == 0)
#endif
      {
        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
      }
    }

    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout->isHzInPML (Hz.getTotalPosition (pos)))
        {
          FieldPointValue* tmp = Hz.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = HzAmplitude.getFieldPointValue (pos);

          FPValue val = tmp->getCurValue ();

          if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    Hz.nextTimeStep ();

    if (usePML)
    {
      Bz.nextTimeStep ();
    }

    ++t;

    if (maxAccuracy < 0)
    {
      is_stable_state = 0;
    }

    DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " PRINTF_MODIFIER ". \n", t, maxAccuracy);
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (t, CURRENT, processId, "2D-TEz-amplitude-Hz");
    dumper.dumpGrid (HzAmplitude, GridCoordinate2D (0), HzAmplitude.getSize ());

    for (int i = 0; i < HzSize.getX (); ++i)
    {
      for (int j = 0; j < HzSize.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (j == HzAmplitude.getSize ().getY () / 2)
        {
          FieldPointValue* tmp = HzAmplitude.getFieldPointValue (pos);

          printf ("%u: " PRINTF_MODIFIER "\n", i, tmp->getCurValue ());
        }
      }
    }
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.");
  }
#endif /* !COMPLEX_FIELD_VALUES */
}

int
SchemeTEz::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

  int is_stable_state = 1;

  FPValue valAmp = amplitudeValue->getCurValue ();

  val = val >= 0 ? val : -val;

  if (val >= valAmp)
  {
    FPValue accuracy = val - valAmp;
    if (valAmp != 0)
    {
      accuracy /= valAmp;
    }
    else if (val != 0)
    {
      accuracy /= val;
    }

    if (accuracy > PhysicsConst::accuracy)
    {
      is_stable_state = 0;

      amplitudeValue->setCurValue (val);
    }

    if (accuracy > *maxAccuracy)
    {
      *maxAccuracy = accuracy;
    }
  }

  return is_stable_state;

#endif /* !COMPLEX_FIELD_VALUES */
}

void
SchemeTEz::performSteps ()
{
#if defined (CUDA_ENABLED)

  ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");

#else /* CUDA_ENABLED */

#ifdef PARALLEL_GRID
  if (calculateAmplitude)
  {
    ASSERT_MESSAGE ("Parallel amplitude mode is not implemented");
  }
#endif /* PARALLEL_GRID */

  performNSteps (0, totalStep);

  if (calculateAmplitude)
  {
    performAmplitudeSteps (totalStep);
  }

#endif /* !CUDA_ENABLED */
}

void
SchemeTEz::initScheme (FPValue dx, FPValue sourceFreq)
{
  sourceFrequency = sourceFreq;
  sourceWaveLength = PhysicsConst::SpeedOfLight / sourceFrequency;

  gridStep = dx;
  courantNum = 2.0;
  gridTimeStep = gridStep / (courantNum * PhysicsConst::SpeedOfLight);
}

void
SchemeTEz::initGrids ()
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  for (int i = 0; i < Eps.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Eps.getSize ().getY (); ++j)
    {
      FieldPointValue* valEps = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valEps->setCurValue (FieldValue (1, 0));
#else /* COMPLEX_FIELD_VALUES */
      valEps->setCurValue (1);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = yeeLayout->getEpsCoordFP (expand (Eps.getTotalPosition (pos))).shrink ();

      GridCoordinateFP2D size = yeeLayout->getEpsCoordFP (expand (Eps.getTotalSize ())).shrink ();

      // if (posAbs.getX () > size.getX () / 2 && posAbs.getX () < yeeLayout->rightBorderTotalField.getX () - 10
      //     && posAbs.getY () > yeeLayout->leftBorderTotalField.getY () + 10 && posAbs.getY () < yeeLayout->rightBorderTotalField.getY () - 10)
      // {
      //   valEps->setCurValue (4.0);
      // }
      if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
          + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
      {
        valEps->setCurValue (4);
      }

      Eps.setFieldPointValue (valEps, pos);
    }
  }

  BMPDumper<GridCoordinate2D> dumper;

  if (dumpRes)
  {
    dumper.init (0, CURRENT, processId, "Eps");
    dumper.dumpGrid (Eps, GridCoordinate2D (0, 0), Eps.getSize ());
  }

  for (int i = 0; i < Mu.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Mu.getSize ().getY (); ++j)
    {
      FieldPointValue* valMu = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valMu->setCurValue (FieldValue (1, 0));
#else /* COMPLEX_FIELD_VALUES */
      valMu->setCurValue (1);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);

      Mu.setFieldPointValue (valMu, pos);
    }
  }

  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  GridCoordinate2D PMLSize = yeeLayout->getLeftBorderPML ().shrink ();

  FPValue boundary = PMLSize.getX () * gridStep;
  uint32_t exponent = 6;
	FPValue R_err = 1e-16;
	FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
	FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

  for (int i = 0; i < SigmaX.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaX.getSize ().getY (); ++j)
    {
      FieldPointValue* valSigma = new FieldPointValue ();

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = yeeLayout->getEpsCoordFP (expand (SigmaX.getTotalPosition (pos))).shrink ();

      GridCoordinateFP2D size = yeeLayout->getEpsCoordFP (expand (SigmaX.getTotalSize ())).shrink ();

      /*
       * FIXME: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getX () < PMLSize.getX ())
      {
        grid_coord dist = PMLSize.getX () - posAbs.getX ();
  			FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
  			FPValue x2 = dist * gridStep;       // lower bounds for point i

  			FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

#ifdef COMPLEX_FIELD_VALUES
  			valSigma->setCurValue (FieldValue (val, 0));
#else /* COMPLEX_FIELD_VALUES */
        valSigma->setCurValue (val);
#endif /* !COMPLEX_FIELD_VALUES */
      }
      else if (posAbs.getX () >= size.getX () - PMLSize.getX ())
      {
        grid_coord dist = posAbs.getX () - (size.getX () - PMLSize.getX ());
  			FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
  			FPValue x2 = dist * gridStep;       // lower bounds for point i

  			//std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
  			FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

#ifdef COMPLEX_FIELD_VALUES
  			valSigma->setCurValue (FieldValue (val, 0));
#else /* COMPLEX_FIELD_VALUES */
        valSigma->setCurValue (val);
#endif /* !COMPLEX_FIELD_VALUES */
      }

      SigmaX.setFieldPointValue (valSigma, pos);
    }
  }

  for (int i = 0; i < SigmaY.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaY.getSize ().getY (); ++j)
    {
      FieldPointValue* valSigma = new FieldPointValue ();

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = yeeLayout->getEpsCoordFP (expand (SigmaY.getTotalPosition (pos))).shrink ();

      GridCoordinateFP2D size = yeeLayout->getEpsCoordFP (expand (SigmaX.getTotalSize ())).shrink ();

      /*
       * FIXME: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getY () < PMLSize.getY ())
      {
        grid_coord dist = PMLSize.getY () - posAbs.getY ();
        FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
        FPValue x2 = dist * gridStep;       // lower bounds for point i

        FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

#ifdef COMPLEX_FIELD_VALUES
  			valSigma->setCurValue (FieldValue (val, 0));
#else /* COMPLEX_FIELD_VALUES */
        valSigma->setCurValue (val);
#endif /* !COMPLEX_FIELD_VALUES */
      }
      else if (posAbs.getY () >= size.getY () - PMLSize.getY ())
      {
        grid_coord dist = posAbs.getY () - (size.getY () - PMLSize.getY ());
        FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
        FPValue x2 = dist * gridStep;       // lower bounds for point i

        //std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
        FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

#ifdef COMPLEX_FIELD_VALUES
  			valSigma->setCurValue (FieldValue (val, 0));
#else /* COMPLEX_FIELD_VALUES */
        valSigma->setCurValue (val);
#endif /* !COMPLEX_FIELD_VALUES */
      }

      SigmaY.setFieldPointValue (valSigma, pos);
    }
  }

  /*
   * FIXME: SigmaZ grid could be replaced with constant 0.0
   */
  for (int i = 0; i < SigmaZ.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaZ.getSize ().getY (); ++j)
    {
      FieldPointValue* valSigma = new FieldPointValue ();

      GridCoordinate2D pos (i, j);

      SigmaZ.setFieldPointValue (valSigma, pos);
    }
  }

  if (dumpRes)
  {
    dumper.init (0, CURRENT, processId, "SigmaX");
    dumper.dumpGrid (SigmaX, GridCoordinate2D (0, 0), SigmaX.getSize ());

    dumper.init (0, CURRENT, processId, "SigmaY");
    dumper.dumpGrid (SigmaY, GridCoordinate2D (0, 0), SigmaY.getSize ());

    dumper.init (0, CURRENT, processId, "SigmaZ");
    dumper.dumpGrid (SigmaZ, GridCoordinate2D (0, 0), SigmaZ.getSize ());
  }

  for (int i = 0; i < Ex.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ex.getSize ().getY (); ++j)
    {
      FieldPointValue* valEx = new FieldPointValue ();

      FieldPointValue* valDx = new FieldPointValue ();

      FieldPointValue* valExAmp;
      if (calculateAmplitude)
      {
        valExAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Ex.setFieldPointValue (valEx, pos);

      Dx.setFieldPointValue (valDx, pos);

      if (calculateAmplitude)
      {
        ExAmplitude.setFieldPointValue (valExAmp, pos);
      }
    }
  }

  for (int i = 0; i < Ey.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ey.getSize ().getY (); ++j)
    {
      FieldPointValue* valEy = new FieldPointValue ();

      FieldPointValue* valDy = new FieldPointValue ();

      FieldPointValue* valEyAmp;
      if (calculateAmplitude)
      {
        valEyAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Ey.setFieldPointValue (valEy, pos);

      Dy.setFieldPointValue (valDy, pos);

      if (calculateAmplitude)
      {
        EyAmplitude.setFieldPointValue (valEyAmp, pos);
      }
    }
  }

  for (int i = 0; i < Hz.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hz.getSize ().getY (); ++j)
    {
      FieldPointValue* valHz = new FieldPointValue ();

      FieldPointValue* valBz = new FieldPointValue ();

      FieldPointValue* valHzAmp;
      if (calculateAmplitude)
      {
        valHzAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Hz.setFieldPointValue (valHz, pos);

      Bz.setFieldPointValue (valBz, pos);

      if (calculateAmplitude)
      {
        HzAmplitude.setFieldPointValue (valHzAmp, pos);
      }
    }
  }

  if (useTFSF)
  {
    for (grid_coord i = 0; i < EInc.getSize ().getX (); ++i)
    {
      FieldPointValue* valE = new FieldPointValue ();

      GridCoordinate1D pos (i);

      EInc.setFieldPointValue (valE, pos);
    }

    for (grid_coord i = 0; i < HInc.getSize ().getX (); ++i)
    {
      FieldPointValue* valH = new FieldPointValue ();

      GridCoordinate1D pos (i);

      HInc.setFieldPointValue (valH, pos);
    }
  }

#if defined (PARALLEL_GRID)
  MPI_Barrier (MPI_COMM_WORLD);
#endif

#if defined (PARALLEL_GRID)
  Eps.share ();
  Mu.share ();

  SigmaX.share ();
  SigmaY.share ();
  SigmaZ.share ();
#endif
}

#endif /* GRID_2D */

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

    FPValue N_lambda = waveLength / gridStep;
    FPValue S = gridTimeStep * PhysicsConst::SpeedOfLight / gridStep;
    FPValue arg = PhysicsConst::Pi * S / N_lambda;

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

    FPValue N_lambda = waveLength / gridStep;
    FPValue S = gridTimeStep * PhysicsConst::SpeedOfLight / gridStep;
    FPValue arg = PhysicsConst::Pi * S / N_lambda;

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

  FPValue freq = PhysicsConst::SpeedOfLight / waveLength;

  GridCoordinate1D pos (0);
  FieldPointValue *valH = HInc.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
  valH->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq),
                                 cos (gridTimeStep * t * 2 * PhysicsConst::Pi * freq)));
#else /* COMPLEX_FIELD_VALUES */
  valH->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));
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

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getExCircuitElement (GridCoordinate3D (pos), LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getExCircuitElement (GridCoordinate3D (pos), LayoutDirection::UP));

      FieldPointValue* valEx = Ex.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (posAbs));
      FieldPointValue* valEps1 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())));

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
        if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

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

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getExCircuitElement (GridCoordinate3D (pos), LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getExCircuitElement (GridCoordinate3D (pos), LayoutDirection::UP));

      FieldPointValue* valDx = Dx.getFieldPointValue (pos);

      FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz2 -= -diff;
        }
      }

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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

      FieldPointValue* valEx = Ex.getFieldPointValue (pos);

      FieldPointValue* valDx = Dx.getFieldPointValue (pos);

      FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
      FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (posAbs));
      FieldPointValue* valEps1 = Eps.getFieldPointValue (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ()))));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ()))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

      FPValue sigmaX = valSigmaX->getCurValue ().real ();
      FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FPValue sigmaX = valSigmaX->getCurValue ();
      FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getEyCircuitElement (GridCoordinate3D (pos), LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getEyCircuitElement (GridCoordinate3D (pos), LayoutDirection::RIGHT));

      FieldPointValue* valEy = Ey.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (posAbs));
      FieldPointValue* valEps1 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())));

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
        if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

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

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getEyCircuitElement (GridCoordinate3D (pos), LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getEyCircuitElement (GridCoordinate3D (pos), LayoutDirection::RIGHT));

      FieldPointValue* valDy = Dy.getFieldPointValue (pos);

      FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz2 -= -diff;
        }
      }

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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

      FieldPointValue* valEy = Ey.getFieldPointValue (pos);

      FieldPointValue* valDy = Dy.getFieldPointValue (pos);

      FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
      FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (posAbs));
      FieldPointValue* valEps1 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())));
      FieldPointValue* valEps2 = Eps.getFieldPointValue (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

      FPValue sigmaX = valSigmaX->getCurValue ().real ();
      FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FPValue sigmaX = valSigmaX->getCurValue ();
      FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::RIGHT));

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::UP));

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);
      FieldPointValue* valMu = Mu.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (posAbs));
      FieldPointValue* valMu1 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu3 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu4 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));

      FieldPointValue* valEy1 = Ey.getFieldPointValue (posRight);
      FieldPointValue* valEy2 = Ey.getFieldPointValue (posLeft);

      FieldPointValue* valEx1 = Ex.getFieldPointValue (posUp);
      FieldPointValue* valEx2 = Ex.getFieldPointValue (posDown);

      FieldValue prevEx1 = valEx1->getPrevValue ();
      FieldValue prevEx2 = valEx2->getPrevValue ();

      FieldValue prevEy1 = valEy1->getPrevValue ();
      FieldValue prevEy2 = valEy2->getPrevValue ();

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real () + valMu3->getCurValue ().real () + valMu4->getCurValue ().real ()) / 4;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;
#endif /* !COMPLEX_FIELD_VALUES */

      if (useTFSF)
      {
        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx2 += diff * sin (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx1 += diff * sin (incidentWaveAngle);
        }

        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy2 += -diff * cos (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ey.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy1 += -diff * cos (incidentWaveAngle);
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

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::RIGHT));

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getHzCircuitElement (pos, LayoutDirection::UP));

      FieldPointValue* valBz = Bz.getFieldPointValue (pos);

      FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);
      FieldPointValue* valMu = Mu.getFieldPointValue (pos);

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
        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx2 += diff * sin (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx1 += diff * sin (incidentWaveAngle);
        }

        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy2 += -diff * cos (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT, false))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ey.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

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

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy1 += -diff * cos (incidentWaveAngle);
        }
      }

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaX = valSigmaX->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaX = valSigmaX->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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

      FieldPointValue* valHz = Hz.getFieldPointValue (pos);

      FieldPointValue* valBz = Bz.getFieldPointValue (pos);

      FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);
      FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (posAbs));
      FieldPointValue* valMu1 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu3 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
      FieldPointValue* valMu4 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real () + valMu3->getCurValue ().real () + valMu4->getCurValue ().real ()) / 4;

      FPValue sigmaY = valSigmaY->getCurValue ().real ();
      FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;

      FPValue sigmaY = valSigmaY->getCurValue ();
      FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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
SchemeTEz::performNSteps (time_step startStep, time_step numberTimeSteps, int dumpRes)
{
  GridCoordinate2D HzSize = Hz.getSize ();

  time_step stepLimit = startStep + numberTimeSteps;

  for (int t = startStep; t < stepLimit; ++t)
  {
    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getEnd ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getEnd ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getEnd ());

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
      if (process == 0)
#endif
      {
        FPValue freq = PhysicsConst::SpeedOfLight / waveLength;

        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * freq)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));
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
        dumperEx.init (t, CURRENT, process, "2D-TEz-in-time-Ex");
        dumperEx.dumpGrid (Ex);

        BMPDumper<GridCoordinate2D> dumperEy;
        dumperEy.init (t, CURRENT, process, "2D-TEz-in-time-Ey");
        dumperEy.dumpGrid (Ey);

        BMPDumper<GridCoordinate2D> dumperHz;
        dumperHz.init (t, CURRENT, process, "2D-TEz-in-time-Hz");
        dumperHz.dumpGrid (Hz);
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

            if (!yeeLayout.isHzInPML (Hz.getTotalPosition (pos)))
            {
              FieldPointValue *value = Hz.getFieldPointValue (pos);

              FieldValue val = value->getCurValue ();

              norm += val.real () * val.real () + val.imag () * val.imag ();
            }
          }
        }

        norm = sqrt (norm);

        printf ("Norm at step %u: %f\n", t, norm);
      }
    }
#endif /* COMPLEX_FIELD_VALUES */
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumperEx;
    dumperEx.init (stepLimit, CURRENT, process, "2D-TEz-in-time-Ex");
    dumperEx.dumpGrid (Ex);

    BMPDumper<GridCoordinate2D> dumperEy;
    dumperEy.init (stepLimit, CURRENT, process, "2D-TEz-in-time-Ey");
    dumperEy.dumpGrid (Ey);

    BMPDumper<GridCoordinate2D> dumperHz;
    dumperHz.init (stepLimit, CURRENT, process, "2D-TEz-in-time-Hz");
    dumperHz.dumpGrid (Hz);
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

        if (!yeeLayout.isHzInPML (Hz.getTotalPosition (pos)))
        {
          FieldPointValue *value = Hz.getFieldPointValue (pos);

          FieldValue val = value->getCurValue ();

          norm += val.real () * val.real () + val.imag () * val.imag ();
        }
      }
    }

    norm = sqrt (norm);

    printf ("Norm at step %u: %f\n", stepLimit, norm);
  }
#endif /* COMPLEX_FIELD_VALUES */
}

void
SchemeTEz::performAmplitudeSteps (time_step startStep, int dumpRes)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

  int is_stable_state = 0;

  GridCoordinate2D HzSize = Hz.getSize ();

  time_step t = startStep;

  GridCoordinate2D PMLSize = shrinkCoord (yeeLayout.getLeftBorderPML ());

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FPValue maxAccuracy = -1;

    //is_stable_state = 1;

    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getEnd ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getEnd ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getEnd ());

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

        if (!yeeLayout.isExInPML (Ex.getTotalPosition (pos)))
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

        if (!yeeLayout.isEyInPML (Ey.getTotalPosition (pos)))
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
      if (process == 0)
#endif
      {
        FPValue freq = PhysicsConst::SpeedOfLight / waveLength;

        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * freq)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));
#endif /* !COMPLEX_FIELD_VALUES */
      }
    }

    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout.isHzInPML (Hz.getTotalPosition (pos)))
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

#if defined (PARALLEL_GRID)
    for (int rank = 0; rank < Hz.getTotalProcCount (); ++rank)
    {
      if (process == rank)
      {
        for (int rankDest = 0; rankDest < Hz.getTotalProcCount (); ++rankDest)
        {
          if (rankDest != rank)
          {
            int retCode = MPI_Send (&is_stable_state, 1, MPI_INT, 0, process, MPI_COMM_WORLD);

            ASSERT (retCode == MPI_SUCCESS);
          }
        }
      }
      else
      {
        MPI_Status status;

        int is_other_stable_state = 0;

        int retCode = MPI_Recv (&is_other_stable_state, 1, MPI_INT, rank, rank, MPI_COMM_WORLD, &status);

        ASSERT (retCode == MPI_SUCCESS);

        if (!is_other_stable_state)
        {
          is_stable_state = 0;
        }
      }

      MPI_Barrier (MPI_COMM_WORLD);
    }
#endif

#if PRINT_MESSAGE
    printf ("%d amplitude calculation step: max accuracy %f. \n", t, maxAccuracy);
#endif /* PRINT_MESSAGE */
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (t, CURRENT, process, "2D-TEz-amplitude-Hz");
    dumper.dumpGrid (HzAmplitude);

    for (int i = 0; i < HzSize.getX (); ++i)
    {
      for (int j = 0; j < HzSize.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (j == HzAmplitude.getSize ().getY () / 2)
        {
          FieldPointValue* tmp = HzAmplitude.getFieldPointValue (pos);

          printf ("%u: %f\n", i, tmp->getCurValue ());
        }
      }
    }
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.\n");
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
SchemeTEz::performSteps (int dumpRes)
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

  performNSteps (0, totalStep, dumpRes);

  if (calculateAmplitude)
  {
    performAmplitudeSteps (totalStep, dumpRes);
  }

#endif /* !CUDA_ENABLED */
}

void
SchemeTEz::initScheme (FPValue wLength, FPValue step)
{
  waveLength = wLength;
  stepWaveLength = step;
  frequency = /*PhysicsConst::SpeedOfLight / waveLength*/ 2.45*1000000000;
  waveLength = PhysicsConst::SpeedOfLight / frequency;

  gridStep = waveLength / stepWaveLength;
  gridTimeStep = 0.4 * gridStep / (PhysicsConst::SpeedOfLight);
}

#if defined (PARALLEL_GRID)
void
SchemeTEz::initProcess (int rank)
{
  process = rank;
}
#endif

void
SchemeTEz::initGrids ()
{
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
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalSize ()));

      // if (posAbs.getX () > size.getX () / 2 && posAbs.getX () < yeeLayout.rightBorderTotalField.getX () - 10
      //     && posAbs.getY () > yeeLayout.leftBorderTotalField.getY () + 10 && posAbs.getY () < yeeLayout.rightBorderTotalField.getY () - 10)
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
  dumper.init (0, CURRENT, process, "Eps");
  dumper.dumpGrid (Eps);

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

  GridCoordinate2D PMLSize = shrinkCoord (yeeLayout.getLeftBorderPML ());

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
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalSize ()));

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
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaY.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalSize ()));

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

  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (0, CURRENT, process, "SigmaX");
    dumper.dumpGrid (SigmaX);
  }

  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (0, CURRENT, process, "SigmaY");
    dumper.dumpGrid (SigmaY);
  }

  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (0, CURRENT, process, "SigmaZ");
    dumper.dumpGrid (SigmaZ);
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
#endif
}

#endif /* GRID_2D */

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

    FieldValue N_lambda = waveLength / gridStep;
    FieldValue S = gridTimeStep * PhysicsConst::SpeedOfLight / gridStep;
    FieldValue arg = PhysicsConst::Pi * S / N_lambda;

    FieldValue relPhi;
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

    FieldValue N_lambda = waveLength / gridStep;
    FieldValue S = gridTimeStep * PhysicsConst::SpeedOfLight / gridStep;
    FieldValue arg = PhysicsConst::Pi * S / N_lambda;

    FieldValue relPhi;
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

  FieldValue freq = PhysicsConst::SpeedOfLight / waveLength;

  GridCoordinate1D pos (0);
  FieldPointValue *valH = HInc.getFieldPointValue (pos);
  valH->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));

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
  FieldValue eps0 = PhysicsConst::Eps0;

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
      FieldValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

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
  FieldValue eps0 = PhysicsConst::Eps0;

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
        if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz2 -= -diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FieldValue k_y = 1;

      FieldValue Ca = (2 * eps0 * k_y - valSigmaY->getCurValue () * gridTimeStep) / (2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep);
      FieldValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep);

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
      FieldValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FieldValue k_x = 1;
      FieldValue k_z = 1;

      FieldValue Ca = (2 * eps0 * k_z - valSigmaZ->getCurValue () * gridTimeStep) / (2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep);
      FieldValue Cb = ((2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep);
      FieldValue Cc = ((2 * eps0 * k_x - valSigmaX->getCurValue () * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep);

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
  FieldValue eps0 = PhysicsConst::Eps0;

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
      FieldValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
      FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

      FieldValue prevHz1 = valHz1->getPrevValue ();
      FieldValue prevHz2 = valHz2->getPrevValue ();

      if (useTFSF)
      {
        if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

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
  FieldValue eps0 = PhysicsConst::Eps0;

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
        if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz1 -= -diff;
        }
        else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (Hz.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle) - 0.5;
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
          FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

          prevHz2 -= -diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FieldValue k_z = 1;

      FieldValue Ca = (2 * eps0 * k_z - valSigmaZ->getCurValue () * gridTimeStep) / (2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep);
      FieldValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep);

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
      FieldValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

      FieldValue k_x = 1;
      FieldValue k_y = 1;

      FieldValue Ca = (2 * eps0 * k_x - valSigmaX->getCurValue () * gridTimeStep) / (2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep);
      FieldValue Cb = ((2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep);
      FieldValue Cc = ((2 * eps0 * k_y - valSigmaY->getCurValue () * gridTimeStep) / (eps * eps0)) / (2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep);

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
  FieldValue mu0 = PhysicsConst::Mu0;

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
      FieldValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;

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
        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx2 += diff * sin (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx1 += diff * sin (incidentWaveAngle);
        }

        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy2 += -diff * cos (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ey.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

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
  FieldValue eps0 = PhysicsConst::Eps0;
  FieldValue mu0 = PhysicsConst::Mu0;

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
        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx2 += diff * sin (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEx1 += diff * sin (incidentWaveAngle);
        }

        if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy2 += -diff * cos (incidentWaveAngle);
        }
        else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT))
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getExCoordFP (Ey.getTotalPosition (posRight)));
          GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

          FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          FieldValue coordD2 = coordD1 + 1;
          FieldValue proportionD2 = d - coordD1;
          FieldValue proportionD1 = 1 - proportionD2;

          GridCoordinate1D pos1 (coordD1);
          GridCoordinate1D pos2 (coordD2);

          FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

          FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

          prevEy1 += -diff * cos (incidentWaveAngle);
        }
      }

      FieldValue k_x = 1;

      FieldValue Ca = (2 * eps0 * k_x - valSigmaX->getCurValue () * gridTimeStep) / (2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep);
      FieldValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_x + valSigmaX->getCurValue () * gridTimeStep);

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
      FieldValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;

      FieldValue k_y = 1;
      FieldValue k_z = 1;

      FieldValue Ca = (2 * eps0 * k_y - valSigmaY->getCurValue () * gridTimeStep) / (2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep);
      FieldValue Cb = ((2 * eps0 * k_z + valSigmaZ->getCurValue () * gridTimeStep) / (mu * mu0)) / (2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep);
      FieldValue Cc = ((2 * eps0 * k_z - valSigmaZ->getCurValue () * gridTimeStep) / (mu * mu0)) / (2 * eps0 * k_y + valSigmaY->getCurValue () * gridTimeStep);

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
        FieldValue freq = PhysicsConst::SpeedOfLight / waveLength;

        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));
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
}

void
SchemeTEz::performAmplitudeSteps (time_step startStep, int dumpRes)
{
  int is_stable_state = 0;

  GridCoordinate2D HzSize = Hz.getSize ();

  time_step t = startStep;

  GridCoordinate2D PMLSize = shrinkCoord (yeeLayout.getLeftBorderPML ());

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FieldValue maxAccuracy = -1;

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

          FieldValue val = tmp->getCurValue ();

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

          FieldValue val = tmp->getCurValue ();

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
        FieldValue freq = PhysicsConst::SpeedOfLight / waveLength;

        GridCoordinate2D pos (HzSize.getX () / 2, HzSize.getY () / 2);
        FieldPointValue* tmp = Hz.getFieldPointValue (pos);
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * freq));
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

          if (updateAmplitude (tmp->getCurValue (), tmpAmp, &maxAccuracy) == 0)
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
}

int
SchemeTEz::updateAmplitude (FieldValue val, FieldPointValue *amplitudeValue, FieldValue *maxAccuracy)
{
  int is_stable_state = 1;

  FieldValue valAmp = amplitudeValue->getCurValue ();

  val = val >= 0 ? val : -val;

  if (val >= valAmp)
  {
    FieldValue accuracy = val - valAmp;
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
}

void
SchemeTEz::performSteps (int dumpRes)
{
#if defined (CUDA_ENABLED)

  ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");

#else /* CUDA_ENABLED */

  performNSteps (0, totalStep, dumpRes);

  if (calculateAmplitude)
  {
    performAmplitudeSteps (totalStep, dumpRes);
  }

#endif /* !CUDA_ENABLED */
}

void
SchemeTEz::initScheme (FieldValue wLength, FieldValue step)
{
  waveLength = wLength;
  stepWaveLength = step;
  frequency = /*PhysicsConst::SpeedOfLight / waveLength*/ 1.25*1000000000;
  waveLength = PhysicsConst::SpeedOfLight / frequency;

  gridStep = waveLength / stepWaveLength;
  gridTimeStep = 0.5 * gridStep / (PhysicsConst::SpeedOfLight);
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
      FieldPointValue* valEps;

#if defined (TWO_TIME_STEPS)
      valEps = new FieldPointValue (1, 1, 1);
#elif defined (ONE_TIME_STEP)
      valEps = new FieldPointValue (1, 1);
#else
      valEps = new FieldPointValue (1);
#endif

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalSize ()));

      // if (posAbs.getX () > size.getX () / 2 && posAbs.getX () < yeeLayout.rightBorderTotalField.getX () - 10
      //     && posAbs.getY () > yeeLayout.leftBorderTotalField.getY () + 10 && posAbs.getY () < yeeLayout.rightBorderTotalField.getY () - 10)
      // {
      //   valEps->setCurValue (4.0);
      // }
      if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2) + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*0.2/0.75) * (size.getX ()*0.2/0.75))
      {
        valEps->setCurValue (58.29);
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
      FieldPointValue* valMu;

#if defined (TWO_TIME_STEPS)
      valMu = new FieldPointValue (1, 1, 1);
#elif defined (ONE_TIME_STEP)
      valMu = new FieldPointValue (1, 1);
#else
      valMu = new FieldPointValue (1);
#endif

      GridCoordinate2D pos (i, j);

      Mu.setFieldPointValue (valMu, pos);
    }
  }

  FieldValue eps0 = PhysicsConst::Eps0;
  FieldValue mu0 = PhysicsConst::Mu0;

  GridCoordinate2D PMLSize = shrinkCoord (yeeLayout.getLeftBorderPML ());

  FieldValue boundary = PMLSize.getX () * gridStep;
  uint32_t exponent = 6;
	FieldValue R_err = 1e-16;
	FieldValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
	FieldValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

  for (int i = 0; i < SigmaX.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaX.getSize ().getY (); ++j)
    {
      FieldPointValue* valSigma;

#if defined (TWO_TIME_STEPS)
      valSigma = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
      valSigma = new FieldPointValue (0, 0);
#else
      valSigma = new FieldPointValue (0);
#endif

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalSize ()));

      /*
       * FIXME: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getX () < PMLSize.getX ())
      {
        grid_coord dist = PMLSize.getX () - posAbs.getX ();
  			FieldValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
  			FieldValue x2 = dist * gridStep;       // lower bounds for point i

  			valSigma->setCurValue (boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1))));   //   polynomial grading
      }
      else if (posAbs.getX () >= size.getX () - PMLSize.getX ())
      {
        grid_coord dist = posAbs.getX () - (size.getX () - PMLSize.getX ());
  			FieldValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
  			FieldValue x2 = dist * gridStep;       // lower bounds for point i

  			//std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
  			valSigma->setCurValue (boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1))));   //   polynomial grading
      }

      SigmaX.setFieldPointValue (valSigma, pos);
    }
  }

  for (int i = 0; i < SigmaY.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaY.getSize ().getY (); ++j)
    {
      FieldPointValue* valSigma;

#if defined (TWO_TIME_STEPS)
      valSigma = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
      valSigma = new FieldPointValue (0, 0);
#else
      valSigma = new FieldPointValue (0);
#endif

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaY.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (SigmaX.getTotalSize ()));

      /*
       * FIXME: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getY () < PMLSize.getY ())
      {
        grid_coord dist = PMLSize.getY () - posAbs.getY ();
        FieldValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
        FieldValue x2 = dist * gridStep;       // lower bounds for point i

        valSigma->setCurValue (boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1))));   //   polynomial grading
      }
      else if (posAbs.getY () >= size.getY () - PMLSize.getY ())
      {
        grid_coord dist = posAbs.getY () - (size.getY () - PMLSize.getY ());
        FieldValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
        FieldValue x2 = dist * gridStep;       // lower bounds for point i

        //std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
        valSigma->setCurValue (boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1))));   //   polynomial grading
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
      FieldPointValue* valSigma;

#if defined (TWO_TIME_STEPS)
      valSigma = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
      valSigma = new FieldPointValue (0, 0);
#else
      valSigma = new FieldPointValue (0);
#endif

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
      FieldPointValue* valExAmp;
      FieldPointValue* valEx;
      FieldPointValue* valDx;

#if defined (TWO_TIME_STEPS)
      valEx = new FieldPointValue (0, 0, 0);

      valDx = new FieldPointValue (0, 0, 0);

      if (calculateAmplitude)
      {
        valExAmp = new FieldPointValue (0, 0, 0);
      }
#elif defined (ONE_TIME_STEP)
      valEx = new FieldPointValue (0, 0);

      valDx = new FieldPointValue (0, 0);

      if (calculateAmplitude)
      {
        valExAmp = new FieldPointValue (0, 0);
      }
#else
      valEx = new FieldPointValue (0);

      valDx = new FieldPointValue (0);

      if (calculateAmplitude)
      {
        valExAmp = new FieldPointValue (0);
      }
#endif

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
      FieldPointValue* valEyAmp;
      FieldPointValue* valEy;
      FieldPointValue* valDy;

#if defined (TWO_TIME_STEPS)
      valEy = new FieldPointValue (0, 0, 0);

      valDy = new FieldPointValue (0, 0, 0);

      if (calculateAmplitude)
      {
        valEyAmp = new FieldPointValue (0, 0, 0);
      }
#elif defined (ONE_TIME_STEP)
      valEy = new FieldPointValue (0, 0);

      valDy = new FieldPointValue (0, 0);

      if (calculateAmplitude)
      {
        valEyAmp = new FieldPointValue (0, 0);
      }
#else
      valEy = new FieldPointValue (0);

      valDy = new FieldPointValue (0);

      if (calculateAmplitude)
      {
        valEyAmp = new FieldPointValue (0);
      }
#endif

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
      FieldPointValue* valHzAmp;
      FieldPointValue* valHz;

      FieldPointValue* valBz;

#if defined (TWO_TIME_STEPS)
      valHz = new FieldPointValue (0, 0, 0);

      valBz = new FieldPointValue (0, 0, 0);

      if (calculateAmplitude)
      {
        valHzAmp = new FieldPointValue (0, 0, 0);
      }
#elif defined (ONE_TIME_STEP)
      valHz = new FieldPointValue (0, 0);

      valBz = new FieldPointValue (0, 0);

      if (calculateAmplitude)
      {
        valHzAmp = new FieldPointValue (0, 0);
      }
#else
      valHz = new FieldPointValue (0);

      valBz = new FieldPointValue (0);

      if (calculateAmplitude)
      {
        valHzAmp = new FieldPointValue (0);
      }
#endif

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
      FieldPointValue* valE;

#if defined (TWO_TIME_STEPS)
      valE = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
      valE = new FieldPointValue (0, 0);
#else
      valE = new FieldPointValue (0);
#endif

      GridCoordinate1D pos (i);

      EInc.setFieldPointValue (valE, pos);
    }

    for (grid_coord i = 0; i < HInc.getSize ().getX (); ++i)
    {
      FieldPointValue* valH;

#if defined (TWO_TIME_STEPS)
      valH = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
      valH = new FieldPointValue (0, 0);
#else
      valH = new FieldPointValue (0);
#endif

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

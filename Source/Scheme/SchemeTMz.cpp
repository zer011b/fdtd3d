#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "Kernels.h"
#include "SchemeTMz.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_2D

void
SchemeTMz::performPlaneWaveESteps (time_step t)
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

  GridCoordinate1D pos (0);
  FieldPointValue *valE = EInc.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
  valE->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                 cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
  valE->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */

  EInc.nextTimeStep ();
}

void
SchemeTMz::performPlaneWaveHSteps (time_step t)
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

  HInc.nextTimeStep ();
}

/*
 * FIXME: replace GridCoordinate3D with GridCoordinate2D
 */
void
SchemeTMz::performEzSteps (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateEzStepPML (t, EzStart, EzEnd);
  }
  else
  {
    calculateEzStep (t, EzStart, EzEnd);
  }
}

void
SchemeTMz::calculateEzStep (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));

      FieldPointValue* valEz = Ez.getFieldPointValue (pos);

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::RIGHT));
      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::UP));

      FieldPointValue* valEps = Eps.getFieldPointValue (Eps.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = valEps->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = valEps->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHy1 = Hy.getFieldPointValue (posRight);
      FieldPointValue* valHy2 = Hy.getFieldPointValue (posLeft);
      FieldPointValue* valHx1 = Hx.getFieldPointValue (posUp);
      FieldPointValue* valHx2 = Hx.getFieldPointValue (posDown);

      FieldValue prevHx1 = valHx1->getPrevValue ();
      FieldValue prevHx2 = valHx2->getPrevValue ();
      FieldValue prevHy1 = valHy1->getPrevValue ();
      FieldValue prevHy2 = valHy2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT, false);

        bool do_need_update_down = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP, false);

        GridCoordinate2D auxPosX;
        GridCoordinate2D auxPosY;
        FieldValue diffX;
        FieldValue diffY;

        if (do_need_update_left)
        {
          auxPosX = posRight;
        }
        else if (do_need_update_right)
        {
          auxPosX = posLeft;
        }

        if (do_need_update_down)
        {
          auxPosY = posUp;
        }
        else if (do_need_update_up)
        {
          auxPosY = posDown;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (Hy.getTotalPosition (auxPosX)));
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

          diffX = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (Hx.getTotalPosition (auxPosY)));
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

          diffY = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevHy1 -= -diffX * cos (incidentWaveAngle);
        }
        else if (do_need_update_right)
        {
          prevHy2 -= -diffX * cos (incidentWaveAngle);
        }

        if (do_need_update_down)
        {
          prevHx1 -= diffY * sin (incidentWaveAngle);
        }
        else if (do_need_update_up)
        {
          prevHx2 -= diffY * sin (incidentWaveAngle);
        }
      }

      FieldValue val = calculateEz_3D (valEz->getPrevValue (),
                                       prevHy1,
                                       prevHy2,
                                       prevHx1,
                                       prevHx2,
                                       gridTimeStep,
                                       gridStep,
                                       eps * eps0);

      valEz->setCurValue (val);
    }
  }
}

void
SchemeTMz::calculateEzStepPML (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));

      FieldPointValue* valDz = Dz.getFieldPointValue (pos);

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::RIGHT));
      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getEzCircuitElement (GridCoordinate3D (pos), LayoutDirection::UP));

      FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaX = valSigmaX->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaX = valSigmaX->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valHy1 = Hy.getFieldPointValue (posRight);
      FieldPointValue* valHy2 = Hy.getFieldPointValue (posLeft);
      FieldPointValue* valHx1 = Hx.getFieldPointValue (posUp);
      FieldPointValue* valHx2 = Hx.getFieldPointValue (posDown);

      FieldValue prevHx1 = valHx1->getPrevValue ();
      FieldValue prevHx2 = valHx2->getPrevValue ();
      FieldValue prevHy1 = valHy1->getPrevValue ();
      FieldValue prevHy2 = valHy2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT, false);

        bool do_need_update_down = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP, false);

        GridCoordinate2D auxPosX;
        GridCoordinate2D auxPosY;
        FieldValue diffX;
        FieldValue diffY;

        if (do_need_update_left)
        {
          auxPosX = posRight;
        }
        else if (do_need_update_right)
        {
          auxPosX = posLeft;
        }

        if (do_need_update_down)
        {
          auxPosY = posUp;
        }
        else if (do_need_update_up)
        {
          auxPosY = posDown;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (Hy.getTotalPosition (auxPosX)));
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

          diffX = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (Hx.getTotalPosition (auxPosY)));
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

          diffY = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevHy1 -= -diffX * cos (incidentWaveAngle);
        }
        else if (do_need_update_right)
        {
          prevHy2 -= -diffX * cos (incidentWaveAngle);
        }

        if (do_need_update_down)
        {
          prevHx1 -= diffY * sin (incidentWaveAngle);
        }
        else if (do_need_update_up)
        {
          prevHx2 -= diffY * sin (incidentWaveAngle);
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_x = 1;

      FPValue Ca = (2 * eps0 * k_x - sigmaX * gridTimeStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);

      FieldValue val = calculateEz_3D_Precalc (valDz->getPrevValue (),
                                               prevHy1,
                                               prevHy2,
                                               prevHx1,
                                               prevHx2,
                                               Ca,
                                               Cb);

      valDz->setCurValue (val);
    }
  }

  if (useMetamaterials)
  {
    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);
        GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
        GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));

        FieldPointValue* valD1z = D1z.getFieldPointValue (pos);
        FieldPointValue* valDz = Dz.getFieldPointValue (pos);

        FieldPointValue* valOmegaPE = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valGammaE = GammaE.getFieldPointValue (GammaE.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

        FieldPointValue* valEps = Eps.getFieldPointValue (Eps.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = valEps->getCurValue ().real ();

        FPValue omegaPE = valOmegaPE->getCurValue ().real ();
        FPValue gammaE = valGammaE->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = valEps->getCurValue ();

        FPValue omegaPE = valOmegaPE->getCurValue ();
        FPValue gammaE = valGammaE->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        /*
         * FIXME: precalculate coefficients
         */
        FPValue A = 4*eps0*eps + 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE;

        FieldValue val = calculateDrudeE (valDz->getCurValue (),
                                          valDz->getPrevValue (),
                                          valDz->getPrevPrevValue (),
                                          valD1z->getPrevValue (),
                                          valD1z->getPrevPrevValue (),
                                          (4 + 2*gridTimeStep*gammaE) / A,
                                          -8 / A,
                                          (4 - 2*gridTimeStep*gammaE) / A,
                                          (2*eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE - 8*eps0*eps) / A,
                                          (4*eps0*eps - 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE) / A);

        valD1z->setCurValue (val);
      }
    }
  }

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));

      FieldPointValue* valEz = Ez.getFieldPointValue (pos);
      FieldPointValue* valDz;

      if (useMetamaterials)
      {
        valDz = D1z.getFieldPointValue (pos);
      }
      else
      {
        valDz = Dz.getFieldPointValue (pos);
      }

      FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valEps = Eps.getFieldPointValue (Eps.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue eps = valEps->getCurValue ().real ();

      FPValue sigmaY = valSigmaY->getCurValue ().real ();
      FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
      FPValue eps = valEps->getCurValue ();

      FPValue sigmaY = valSigmaY->getCurValue ();
      FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      FPValue modifier = eps * eps0;
      if (useMetamaterials)
      {
        modifier = 1;
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_y = 1;
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_z + sigmaZ * gridTimeStep) / (modifier)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_z - sigmaZ * gridTimeStep) / (modifier)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

      FieldValue val = calculateEz_from_Dz_Precalc (valEz->getPrevValue (),
                                                    valDz->getCurValue (),
                                                    valDz->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valEz->setCurValue (val);
    }
  }
}

void
SchemeTMz::performHxSteps (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateHxStepPML (t, HxStart, HxEnd);
  }
  else
  {
    calculateHxStep (t, HxStart, HxEnd);
  }
}

void
SchemeTMz::calculateHxStep (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hx.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (posAbs));

      FieldPointValue* valHx = Hx.getFieldPointValue (pos);

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getHxCircuitElement (pos, LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getHxCircuitElement (pos, LayoutDirection::UP));

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEz1 = Ez.getFieldPointValue (posUp);
      FieldPointValue* valEz2 = Ez.getFieldPointValue (posDown);

      FieldValue prevEz1 = valEz1->getPrevValue ();
      FieldValue prevEz2 = valEz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_down = yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_down)
        {
          auxPos = posDown;
        }
        else if (do_need_update_up)
        {
          auxPos = posUp;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (Ez.getTotalPosition (auxPos)));
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

          diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevEz2 += diff;
        }
        else if (do_need_update_up)
        {
          prevEz1 += diff;
        }
      }

      FieldValue val = calculateHx_2D_TMz (valHx->getPrevValue (),
                                           prevEz1,
                                           prevEz2,
                                           gridTimeStep,
                                           gridStep,
                                           mu * mu0);

      valHx->setCurValue (val);
    }
  }
}

void
SchemeTMz::calculateHxStepPML (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hx.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (posAbs));

      FieldPointValue* valBx = Bx.getFieldPointValue (pos);

      GridCoordinate2D posDown = shrinkCoord (yeeLayout.getHxCircuitElement (pos, LayoutDirection::DOWN));
      GridCoordinate2D posUp = shrinkCoord (yeeLayout.getHxCircuitElement (pos, LayoutDirection::UP));

      FieldPointValue* valSigmaY1 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaY2 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaY = (valSigmaY1->getCurValue ().real () + valSigmaY2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaY = (valSigmaY1->getCurValue () + valSigmaY2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEz1 = Ez.getFieldPointValue (posUp);
      FieldPointValue* valEz2 = Ez.getFieldPointValue (posDown);

      FieldValue prevEz1 = valEz1->getPrevValue ();
      FieldValue prevEz2 = valEz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_down = yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN, false);
        bool do_need_update_up = yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_down)
        {
          auxPos = posDown;
        }
        else if (do_need_update_up)
        {
          auxPos = posUp;
        }

        if (do_need_update_down || do_need_update_up)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (Ez.getTotalPosition (auxPos)));
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

          diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_down)
        {
          prevEz2 += diff;
        }
        else if (do_need_update_up)
        {
          prevEz1 += diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_y = 1;

      FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

      FieldValue val = calculateHx_2D_TMz_Precalc (valBx->getPrevValue (),
                                                   prevEz1,
                                                   prevEz2,
                                                   Ca,
                                                   Cb);

      valBx->setCurValue (val);
    }
  }

  if (useMetamaterials)
  {
    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);
        GridCoordinate2D posAbs = Hx.getTotalPosition (pos);
        GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (posAbs));

        FieldPointValue* valB1x = B1x.getFieldPointValue (pos);
        FieldPointValue* valBx = Bx.getFieldPointValue (pos);

        FieldPointValue* valOmegaPM1 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valOmegaPM2 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valGammaM1 = GammaM.getFieldPointValue (GammaM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valGammaM2 = GammaM.getFieldPointValue (GammaM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

        FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

        FPValue omegaPM;
        FPValue gammaM;
        FPValue dividerOmega = 0;
        FPValue dividerGamma = 0;

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;

        FPValue omegaPM1 = valOmegaPM1->getCurValue ().real ();
        FPValue omegaPM2 = valOmegaPM2->getCurValue ().real ();

        FPValue gammaM1 = valGammaM1->getCurValue ().real ();
        FPValue gammaM2 = valGammaM2->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;

        FPValue omegaPM1 = valOmegaPM1->getCurValue ();
        FPValue omegaPM2 = valOmegaPM2->getCurValue ();

        FPValue gammaM1 = valGammaM1->getCurValue ();
        FPValue gammaM2 = valGammaM2->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        if (omegaPM1 == 0
            || omegaPM2 == 0)
        {
          dividerOmega = sqrtf (2.0);
          dividerGamma = 2.0;
        }
        else
        {
          if (omegaPM1 != omegaPM2
              || gammaM1 != gammaM2)
          {
            ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
          }

          dividerOmega = 2.0;
          dividerGamma = 2.0;
        }

        ASSERT (dividerOmega != 0);
        ASSERT (dividerGamma != 0);

        omegaPM = (omegaPM1 + omegaPM2) / dividerOmega;
        gammaM = (gammaM1 + gammaM2) / dividerGamma;

        /*
         * FIXME: precalculate coefficients
         */
        FPValue C = 4*mu0*mu + 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM;

        FieldValue val = calculateDrudeH (valBx->getCurValue (),
                                          valBx->getPrevValue (),
                                          valBx->getPrevPrevValue (),
                                          valB1x->getPrevValue (),
                                          valB1x->getPrevPrevValue (),
                                          (4 + 2*gridTimeStep*gammaM) / C,
                                          -8 / C,
                                          (4 - 2*gridTimeStep*gammaM) / C,
                                          (2*mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM - 8*mu0*mu) / C,
                                          (4*mu0*mu - 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM) / C);

        valB1x->setCurValue (val);
      }
    }
  }

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hx.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHxCoordFP (posAbs));

      FieldPointValue* valHx = Hx.getFieldPointValue (pos);
      FieldPointValue* valBx;

      if (useMetamaterials)
      {
        valBx = B1x.getFieldPointValue (pos);
      }
      else
      {
        valBx = Bx.getFieldPointValue (pos);
      }

      FieldPointValue* valSigmaX1 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaX2 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

      FieldPointValue* valSigmaZ1 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaZ2 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () + 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX (), realCoord.getY () - 0.5, yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue ().real () + valSigmaX2->getCurValue ().real ()) / 2;
      FPValue sigmaZ = (valSigmaZ1->getCurValue ().real () + valSigmaZ2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue () + valSigmaX2->getCurValue ()) / 2;
      FPValue sigmaZ = (valSigmaZ1->getCurValue () + valSigmaZ2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FPValue modifier = mu * mu0;
      if (useMetamaterials)
      {
        modifier = 1;
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_x = 1;
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_x + sigmaX * gridTimeStep) / (modifier)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_x - sigmaX * gridTimeStep) / (modifier)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

      FieldValue val = calculateHx_from_Bx_Precalc (valHx->getPrevValue (),
                                                    valBx->getCurValue (),
                                                    valBx->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valHx->setCurValue (val);
    }
  }
}

void
SchemeTMz::performHySteps (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (usePML)
  {
    calculateHyStepPML (t, HyStart, HyEnd);
  }
  else
  {
    calculateHyStep (t, HyStart, HyEnd);
  }
}

void
SchemeTMz::calculateHyStep (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hy.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (posAbs));

      FieldPointValue* valHy = Hy.getFieldPointValue (pos);

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getHyCircuitElement (pos, LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getHyCircuitElement (pos, LayoutDirection::RIGHT));

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEz1 = Ez.getFieldPointValue (posRight);
      FieldPointValue* valEz2 = Ez.getFieldPointValue (posLeft);

      FieldValue prevEz1 = valEz1->getPrevValue ();
      FieldValue prevEz2 = valEz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_left)
        {
          auxPos = posLeft;
        }
        else if (do_need_update_right)
        {
          auxPos = posRight;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (Ez.getTotalPosition (auxPos)));
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

          diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevEz2 += diff;
        }
        else if (do_need_update_right)
        {
          prevEz1 += diff;
        }
      }

      FieldValue val = calculateHy_2D_TMz (valHy->getPrevValue (),
                                           prevEz1,
                                           prevEz2,
                                           gridTimeStep,
                                           gridStep,
                                           mu * mu0);

      valHy->setCurValue (val);
    }
  }
}

void
SchemeTMz::calculateHyStepPML (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hy.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (posAbs));

      FieldPointValue* valBy = By.getFieldPointValue (pos);

      GridCoordinate2D posLeft = shrinkCoord (yeeLayout.getHyCircuitElement (pos, LayoutDirection::LEFT));
      GridCoordinate2D posRight = shrinkCoord (yeeLayout.getHyCircuitElement (pos, LayoutDirection::RIGHT));

      FieldPointValue* valSigmaZ1 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaZ2 = SigmaZ.getFieldPointValue (SigmaZ.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue sigmaZ = (valSigmaZ1->getCurValue ().real () + valSigmaZ2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue sigmaZ = (valSigmaZ1->getCurValue () + valSigmaZ2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FieldPointValue* valEz1 = Ez.getFieldPointValue (posRight);
      FieldPointValue* valEz2 = Ez.getFieldPointValue (posLeft);

      FieldValue prevEz1 = valEz1->getPrevValue ();
      FieldValue prevEz2 = valEz2->getPrevValue ();

      if (useTFSF)
      {
        bool do_need_update_left = yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT, false);
        bool do_need_update_right = yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT, false);

        GridCoordinate2D auxPos;
        FieldValue diff;

        if (do_need_update_left)
        {
          auxPos = posLeft;
        }
        else if (do_need_update_right)
        {
          auxPos = posRight;
        }

        if (do_need_update_left || do_need_update_right)
        {
          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (Ez.getTotalPosition (auxPos)));
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

          diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
        }

        if (do_need_update_left)
        {
          prevEz2 += diff;
        }
        else if (do_need_update_right)
        {
          prevEz1 += diff;
        }
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_z = 1;

      FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
      FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

      FieldValue val = calculateHy_2D_TMz_Precalc (valBy->getPrevValue (),
                                                   prevEz1,
                                                   prevEz2,
                                                   Ca,
                                                   Cb);

      valBy->setCurValue (val);
    }
  }

  if (useMetamaterials)
  {
    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);
        GridCoordinate2D posAbs = Hy.getTotalPosition (pos);
        GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (posAbs));

        FieldPointValue* valB1y = B1y.getFieldPointValue (pos);
        FieldPointValue* valBy = By.getFieldPointValue (pos);

        FieldPointValue* valOmegaPM1 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valOmegaPM2 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valGammaM1 = GammaM.getFieldPointValue (GammaM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valGammaM2 = GammaM.getFieldPointValue (GammaM.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

        FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
        FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

        FPValue omegaPM;
        FPValue gammaM;
        FPValue dividerOmega = 0;
        FPValue dividerGamma = 0;

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;

        FPValue omegaPM1 = valOmegaPM1->getCurValue ().real ();
        FPValue omegaPM2 = valOmegaPM2->getCurValue ().real ();

        FPValue gammaM1 = valGammaM1->getCurValue ().real ();
        FPValue gammaM2 = valGammaM2->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;

        FPValue omegaPM1 = valOmegaPM1->getCurValue ();
        FPValue omegaPM2 = valOmegaPM2->getCurValue ();

        FPValue gammaM1 = valGammaM1->getCurValue ();
        FPValue gammaM2 = valGammaM2->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        if (omegaPM1 == 0
            || omegaPM2 == 0)
        {
          dividerOmega = sqrtf (2.0);
          dividerGamma = 2.0;
        }
        else
        {
          if (omegaPM1 != omegaPM2
              || gammaM1 != gammaM2)
          {
            ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
          }

          dividerOmega = 2.0;
          dividerGamma = 2.0;
        }

        ASSERT (dividerOmega != 0);
        ASSERT (dividerGamma != 0);

        omegaPM = (omegaPM1 + omegaPM2) / dividerOmega;
        gammaM = (gammaM1 + gammaM2) / dividerGamma;

        /*
         * FIXME: precalculate coefficients
         */
        FPValue C = 4*mu0*mu + 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM;

        FieldValue val = calculateDrudeH (valBy->getCurValue (),
                                          valBy->getPrevValue (),
                                          valBy->getPrevPrevValue (),
                                          valB1y->getPrevValue (),
                                          valB1y->getPrevPrevValue (),
                                          (4 + 2*gridTimeStep*gammaM) / C,
                                          -8 / C,
                                          (4 - 2*gridTimeStep*gammaM) / C,
                                          (2*mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM - 8*mu0*mu) / C,
                                          (4*mu0*mu - 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM) / C);

        valB1y->setCurValue (val);
      }
    }
  }

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      GridCoordinate2D pos (i, j);
      GridCoordinate2D posAbs = Hy.getTotalPosition (pos);
      GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHyCoordFP (posAbs));

      FieldPointValue* valHy = Hy.getFieldPointValue (pos);
      FieldPointValue* valBy;

      if (useMetamaterials)
      {
        valBy = B1y.getFieldPointValue (pos);
      }
      else
      {
        valBy = By.getFieldPointValue (pos);
      }

      FieldPointValue* valSigmaX1 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaX2 = SigmaX.getFieldPointValue (SigmaX.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

      FieldPointValue* valSigmaY1 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valSigmaY2 = SigmaY.getFieldPointValue (SigmaY.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

      FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));
      FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (shrinkCoord (yeeLayout.getEpsCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY (), yeeLayout.getMinEpsCoordFP ().getZ ())))));

#ifdef COMPLEX_FIELD_VALUES
      FPValue mu = (valMu1->getCurValue ().real () + valMu2->getCurValue ().real ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue ().real () + valSigmaX2->getCurValue ().real ()) / 2;
      FPValue sigmaY = (valSigmaY1->getCurValue ().real () + valSigmaY2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
      FPValue mu = (valMu1->getCurValue () + valMu2->getCurValue ()) / 2;

      FPValue sigmaX = (valSigmaX1->getCurValue () + valSigmaX2->getCurValue ()) / 2;
      FPValue sigmaY = (valSigmaY1->getCurValue () + valSigmaY2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

      FPValue modifier = mu * mu0;
      if (useMetamaterials)
      {
        modifier = 1;
      }

      /*
       * FIXME: precalculate coefficients
       */
      FPValue k_x = 1;
      FPValue k_y = 1;

      FPValue Ca = (2 * eps0 * k_x - sigmaX * gridTimeStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cb = ((2 * eps0 * k_y + sigmaY * gridTimeStep) / (modifier)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
      FPValue Cc = ((2 * eps0 * k_y - sigmaY * gridTimeStep) / (modifier)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);

      FieldValue val = calculateHy_from_By_Precalc (valHy->getPrevValue (),
                                                    valBy->getCurValue (),
                                                    valBy->getPrevValue (),
                                                    Ca,
                                                    Cb,
                                                    Cc);

      valHy->setCurValue (val);
    }
  }
}

void
SchemeTMz::performNSteps (time_step startStep, time_step numberTimeSteps, int dumpRes)
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  GridCoordinate2D EzSize = Ez.getSize ();

  time_step stepLimit = startStep + numberTimeSteps;

  for (int t = startStep; t < stepLimit; ++t)
  {
    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getComputationStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getComputationEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getComputationStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getComputationEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getComputationStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getComputationEnd ());

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performEzSteps (t, EzStart, EzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      if (processId == 0)
#endif
      {
        GridCoordinate2D pos (70, EzSize.getY () / 2);
        FieldPointValue* tmp = Ez.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */

        // for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
        // {
        //   for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
        //   {
        //     GridCoordinate2D pos (i, j);
        //
        //     GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
        //
        //     GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));
        //
        //     if (realCoord.getX () == EzSize.getX () / 2 - 20 + 0.5
        //         || realCoord.getX () == EzSize.getX () / 2 + 20 + 0.5
        //         || realCoord.getY () == EzSize.getY () / 2 - 20 + 0.5
        //         || realCoord.getY () == EzSize.getY () / 2 + 20 + 0.5)
        //     {
        //       FieldPointValue* tmp = Ez.getFieldPointValue (pos);
        //
        //       FieldValue diff_x = (i - ((FieldValue) EzSize.getX ()) / 2);
        //       FieldValue diff_y = (j - ((FieldValue) EzSize.getY ()) / 2);
        //
        //       FieldValue sqr = diff_x * diff_x + diff_y * diff_y;
        //
        //     // if (sqr >= 100 && sqr < 400)
        //     // {
        //       FieldValue inTime = gridTimeStep * t * 2 * PhysicsConst::Pi * freq;
        //       FieldValue inSpace = 2 * PhysicsConst::Pi * sqrt (sqr) / stepWaveLength;
        //
        //       if (gridTimeStep * t * PhysicsConst::SpeedOfLight - gridStep * sqrt (sqr) >= 0)
        //       {
        //         tmp->setCurValue (sin (inTime - inSpace));
        //       }
        //     }
        //   }
        // }
      }
    }

    Ez.nextTimeStep ();

    if (usePML)
    {
      Dz.nextTimeStep ();
    }

    if (useMetamaterials)
    {
      D1z.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();

    if (usePML)
    {
      Bx.nextTimeStep ();
      By.nextTimeStep ();
    }

    if (useMetamaterials)
    {
      B1x.nextTimeStep ();
      B1y.nextTimeStep ();
    }

    if (t % 100 == 0)
    {
      if (dumpRes)
      {
        BMPDumper<GridCoordinate2D> dumperEz;
        dumperEz.init (t, CURRENT, processId, "2D-TMz-in-time-Ez");
        dumperEz.dumpGrid (Ez);

        BMPDumper<GridCoordinate2D> dumperHx;
        dumperHx.init (t, CURRENT, processId, "2D-TMz-in-time-Hx");
        dumperHx.dumpGrid (Hx);

        BMPDumper<GridCoordinate2D> dumperHy;
        dumperHy.init (t, CURRENT, processId, "2D-TMz-in-time-Hy");
        dumperHy.dumpGrid (Hy);
      }
    }
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumperEz;
    dumperEz.init (stepLimit, CURRENT, processId, "2D-TMz-in-time-Ez");
    dumperEz.dumpGrid (Ez);

    BMPDumper<GridCoordinate2D> dumperHx;
    dumperHx.init (stepLimit, CURRENT, processId, "2D-TMz-in-time-Hx");
    dumperHx.dumpGrid (Hx);

    BMPDumper<GridCoordinate2D> dumperHy;
    dumperHy.init (stepLimit, CURRENT, processId, "2D-TMz-in-time-Hy");
    dumperHy.dumpGrid (Hy);

    // for (int i = 0; i < EzSize.getX (); ++i)
    // {
    //   for (int j = 0; j < EzSize.getY (); ++j)
    //   {
    //     GridCoordinate2D pos (i, j);
    //
    //     if (j == Ez.getSize ().getY () / 2)
    //     {
    //       FieldPointValue* tmp = Ez.getFieldPointValue (pos);
    //
    //       FieldValue freq = PhysicsConst::SpeedOfLight / waveLength;
    //       FieldValue inTime = gridTimeStep * (stepLimit - 1) * 2 * PhysicsConst::Pi * freq;
    //       FieldValue inSpace = 2 * PhysicsConst::Pi * (i >= EzSize.getX ()/2 ? i - EzSize.getX ()/2 : EzSize.getX ()/2 - i) / stepWaveLength;
    //
    //       printf ("%u: %f vs %f\n", i, tmp->getCurValue (), sin (inTime - inSpace));
    //     }
    //   }
    // }
  }
}

void
SchemeTMz::performAmplitudeSteps (time_step startStep, int dumpRes)
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

  GridCoordinate2D EzSize = Ez.getSize ();

  time_step t = startStep;

  GridCoordinate2D PMLSize = shrinkCoord (yeeLayout.getLeftBorderPML ());

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FPValue maxAccuracy = -1;

    //is_stable_state = 1;

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getComputationStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getComputationEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getComputationStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getComputationEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getComputationStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getComputationEnd ());

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performEzSteps (t, EzStart, EzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      if (processId == 0)
#endif
      {
        GridCoordinate2D pos (70, EzSize.getY () / 2);
        FieldPointValue* tmp = Ez.getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
        tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                      cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
        tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */

        // for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
        // {
        //   for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
        //   {
        //     GridCoordinate2D pos (i, j);
        //
        //     GridCoordinate2D posAbs = Ez.getTotalPosition (pos);
        //
        //     GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (posAbs));
        //
        //     if (realCoord.getX () == EzSize.getX () / 2 - 20 + 0.5
        //         || realCoord.getX () == EzSize.getX () / 2 + 20 + 0.5
        //         || realCoord.getY () == EzSize.getY () / 2 - 20 + 0.5
        //         || realCoord.getY () == EzSize.getY () / 2 + 20 + 0.5)
        //     {
        //       FieldPointValue* tmp = Ez.getFieldPointValue (pos);
        //
        //       FieldValue diff_x = (i - ((FieldValue) EzSize.getX ()) / 2);
        //       FieldValue diff_y = (j - ((FieldValue) EzSize.getY ()) / 2);
        //
        //       FieldValue sqr = diff_x * diff_x + diff_y * diff_y;
        //
        //       FieldValue inTime = gridTimeStep * t * 2 * PhysicsConst::Pi * freq;
        //       FieldValue inSpace = 2 * PhysicsConst::Pi * sqrt (sqr) / stepWaveLength;
        //
        //     // if (sqr == 0 && )
        //     // {
        //     //   tmp->setCurValue (sin (inTime - inSpace));
        //     // }
        //
        //     // if (sqr >= 100 && sqr < 400)
        //     // {
        //
        //
        //       if (gridTimeStep * t * PhysicsConst::SpeedOfLight - gridStep * sqrt (sqr) >= 0)
        //       {
        //         tmp->setCurValue (sin (inTime - inSpace));
        //       }
        //     }
        //   }
        // }
      }
    }

    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout.isEzInPML (Ez.getTotalPosition (pos)))
        {
          FieldPointValue* tmp = Ez.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = EzAmplitude.getFieldPointValue (pos);

          GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getEzCoordFP (Ez.getTotalPosition (pos)));

          GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
          GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

          FPValue val = tmp->getCurValue ();

          // if (realCoord.getX () > leftBorder.getX () - 1 && realCoord.getX () < rightBorder.getX () + 1
          //     && realCoord.getY () > leftBorder.getY () - 1 && realCoord.getY () < rightBorder.getY () + 1)
          // {
          //   GridCoordinateFP2D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();
          //
          //   FieldValue x = realCoord.getX () - zeroCoordFP.getX ();
          //   FieldValue y = realCoord.getY () - zeroCoordFP.getY ();
          //   FieldValue d = x * cos (incidentWaveAngle) + y * sin (incidentWaveAngle);
          //   FieldValue coordD1 = (FieldValue) ((grid_iter) d);
          //   FieldValue coordD2 = coordD1 + 1;
          //   FieldValue proportionD2 = d - coordD1;
          //   FieldValue proportionD1 = 1 - proportionD2;
          //
          //   GridCoordinate1D pos1 (coordD1);
          //   GridCoordinate1D pos2 (coordD2);
          //
          //   FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
          //   FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);
          //
          //   FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();
          //
          //   val -= diff;
          // }

          if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    Ez.nextTimeStep ();

    if (usePML)
    {
      Dz.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);

    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout.isHxInPML (Hx.getTotalPosition (pos)))
        {
          FieldPointValue* tmp = Hx.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = HxAmplitude.getFieldPointValue (pos);

          if (updateAmplitude (tmp->getCurValue (), tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        if (!yeeLayout.isHyInPML (Hy.getTotalPosition (pos)))
        {

          FieldPointValue* tmp = Hy.getFieldPointValue (pos);
          FieldPointValue* tmpAmp = HyAmplitude.getFieldPointValue (pos);

          if (updateAmplitude (tmp->getCurValue (), tmpAmp, &maxAccuracy) == 0)
          {
            is_stable_state = 0;
          }
        }
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();

    if (usePML)
    {
      Bx.nextTimeStep ();
      By.nextTimeStep ();
    }

    ++t;

    if (maxAccuracy < 0)
    {
      is_stable_state = 0;
    }

#if PRINT_MESSAGE
    printf ("%d amplitude calculation step: max accuracy %f. \n", t, maxAccuracy);
#endif /* PRINT_MESSAGE */
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (t, CURRENT, processId, "2D-TMz-amplitude-Ez");
    dumper.dumpGrid (EzAmplitude);

    // FieldValue norm = 0;
    //
    // for (int i = 0; i < EzSize.getX (); ++i)
    // {
    //   for (int j = 0; j < EzSize.getY (); ++j)
    //   {
    //     GridCoordinate2D pos (i, j);
    //
    //     if (j == EzAmplitude.getSize ().getY () / 2)
    //     {
    //       FieldPointValue* tmp = EzAmplitude.getFieldPointValue (pos);
    //
    //       FieldValue val1 = j0 (2 * PhysicsConst::Pi * (i >= EzSize.getX ()/2 ? i - EzSize.getX ()/2 : EzSize.getX ()/2 - i) / stepWaveLength);
    //       FieldValue val2 = y0 (2 * PhysicsConst::Pi * (i >= EzSize.getX ()/2 ? i - EzSize.getX ()/2 : EzSize.getX ()/2 - i) / stepWaveLength);
    //       FieldValue hankel =  sqrt (val1 * val1 + val2 * val2) / 4;
    //
    //       if (i > EzSize.getX ()/2 && i < yeeLayout.getRightBorderPML ().getX () - 5)
    //       {
    //         norm += tmp->getCurValue () * tmp->getCurValue ();
    //       }
    //
    //       printf ("%u: %f, %f, %f\n", i, tmp->getCurValue (), hankel, tmp->getCurValue () - hankel);
    //     }
    //   }
    // }

    // norm = sqrt (norm);
    //
    // printf ("Norm (C): %f\n", norm);
    //
    // for (int i = 0; i < EzSize.getX (); ++i)
    // {
    //   for (int j = 0; j < EzSize.getY (); ++j)
    //   {
    //     GridCoordinate2D pos (i, j);
    //
    //     if (j == EzAmplitude.getSize ().getY () / 2)
    //     {
    //       FieldPointValue* tmp = EzAmplitude.getFieldPointValue (pos);
    //
    //       printf ("%u: %f\n", i, tmp->getCurValue () / norm);
    //     }
    //   }
    // }
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.\n");
  }
#endif /* !COMPLEX_FIELD_VALUES */
}

int
SchemeTMz::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
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
SchemeTMz::performSteps (int dumpRes)
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

#if defined (CUDA_ENABLED)

  if (usePML || useTFSF || calculateAmplitude || useMetamaterials)
  {
    ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");
  }

  CudaExitStatus status;

  cudaExecute2DTMzSteps (&status, yeeLayout, gridTimeStep, gridStep, Ez, Hx, Hy, Eps, Mu, totalStep, processId);

  ASSERT (status == CUDA_OK);

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (totalStep, ALL, processId, "2D-TMz-in-time");
    dumper.dumpGrid (Ez);
  }
#else /* CUDA_ENABLED */

  if (useMetamaterials && !usePML)
  {
    ASSERT_MESSAGE ("Metamaterials without pml are not implemented");
  }

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
SchemeTMz::initScheme (FPValue dx, FPValue sourceFreq)
{
  sourceFrequency = sourceFreq;
  sourceWaveLength = PhysicsConst::SpeedOfLight / sourceFrequency;

  gridStep = dx;
  courantNum = 2.0;
  gridTimeStep = gridStep / (courantNum * PhysicsConst::SpeedOfLight);
}

void
SchemeTMz::initGrids ()
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
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalSize ()));

      // if (posAbs.getX () > size.getX () / 2 && posAbs.getX () < yeeLayout.rightBorderTotalField.getX () - 10
      //     && posAbs.getY () > yeeLayout.leftBorderTotalField.getY () + 10 && posAbs.getY () < yeeLayout.rightBorderTotalField.getY () - 10)
      // {
      //   valEps->setCurValue (4.0);
      // }
      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
      // {
      //   valEps->setCurValue (4);
      // }

//       if (posAbs.getX () >= size.getX () / 2 - 32 && posAbs.getX () < size.getX () / 2 + 32
//           && posAbs.getY () >= 0 && posAbs.getY () < size.getY ())
//       {
// #ifdef COMPLEX_FIELD_VALUES
//         valEps->setCurValue (FieldValue (4, 0));
// #else /* COMPLEX_FIELD_VALUES */
//         valEps->setCurValue (4);
// #endif /* !COMPLEX_FIELD_VALUES */
//       }

      Eps.setFieldPointValue (valEps, pos);
    }
  }

  BMPDumper<GridCoordinate2D> dumper;
  dumper.init (0, CURRENT, processId, "Eps");
  dumper.dumpGrid (Eps);

  for (int i = 0; i < OmegaPE.getSize ().getX (); ++i)
  {
    for (int j = 0; j < OmegaPE.getSize ().getY (); ++j)
    {
      FieldPointValue* valOmega = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valOmega->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
      valOmega->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (OmegaPE.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (OmegaPE.getTotalSize ()));

      if (posAbs.getX () >= 120 && posAbs.getX () < size.getX () - 120
          && posAbs.getY () >= yeeLayout.getLeftBorderPML ().getY () && posAbs.getY () < size.getY () - yeeLayout.getLeftBorderPML ().getY ())
      {

      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0)
      //     && (posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //         + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) > (size.getX ()*0.5/7.0) * (size.getX ()*0.5/7.0))
      // {
#ifdef COMPLEX_FIELD_VALUES
        valOmega->setCurValue (FieldValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency, 0));
#else /* COMPLEX_FIELD_VALUES */
        valOmega->setCurValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency);
#endif /* !COMPLEX_FIELD_VALUES */
      }

      OmegaPE.setFieldPointValue (valOmega, pos);
    }
  }

  for (int i = 0; i < OmegaPM.getSize ().getX (); ++i)
  {
    for (int j = 0; j < OmegaPM.getSize ().getY (); ++j)
    {
      FieldPointValue* valOmega = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valOmega->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
      valOmega->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);
      GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (OmegaPM.getTotalPosition (pos)));

      GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (OmegaPM.getTotalSize ()));

      // if (posAbs.getX () >= size.getX () / 2 - 20 && posAbs.getX () < size.getX () / 2 + 20
      //     && posAbs.getY () >= 50 && posAbs.getY () < size.getY () - 50)s
      // {
      //   valOmega->setCurValue (sqrt(2) * 2 * PhysicsConst::Pi * frequency);
      // }

      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
      // {
      //   valOmega->setCurValue (1);
      // }

      if (posAbs.getX () >= 120 && posAbs.getX () < size.getX () - 120
          && posAbs.getY () >= yeeLayout.getLeftBorderPML ().getY () && posAbs.getY () < size.getY () - yeeLayout.getLeftBorderPML ().getY ())
      {

      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0)
      //     && (posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //         + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) > (size.getX ()*0.5/7.0) * (size.getX ()*0.5/7.0))
      // {
#ifdef COMPLEX_FIELD_VALUES
        valOmega->setCurValue (FieldValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency, 0));
#else /* COMPLEX_FIELD_VALUES */
        valOmega->setCurValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency);
#endif /* !COMPLEX_FIELD_VALUES */
      }

      OmegaPM.setFieldPointValue (valOmega, pos);
    }
  }

  for (int i = 0; i < GammaE.getSize ().getX (); ++i)
  {
    for (int j = 0; j < GammaE.getSize ().getY (); ++j)
    {
      FieldPointValue* valGamma = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valGamma->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
      valGamma->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);
      // GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalPosition (pos)));
      //
      // GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (Eps.getTotalSize ()));

      // GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (GammaE.getTotalPosition (pos)));
      //
      // GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (GammaE.getTotalSize ()));
      //
      // if (posAbs.getX () >= size.getX () / 2 - 20 && posAbs.getX () < size.getX () / 2 + 20
      //     && posAbs.getY () >= 50 && posAbs.getY () < size.getY () - 50)
      // {
      //   valGamma->setCurValue (1);
      // }

      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
      // {
      //   valGamma->setCurValue (1);
      // }

      GammaE.setFieldPointValue (valGamma, pos);
    }
  }

  for (int i = 0; i < GammaM.getSize ().getX (); ++i)
  {
    for (int j = 0; j < GammaM.getSize ().getY (); ++j)
    {
      FieldPointValue* valGamma = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
      valGamma->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
      valGamma->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

      GridCoordinate2D pos (i, j);
      // GridCoordinateFP2D posAbs = shrinkCoord (yeeLayout.getEpsCoordFP (GammaM.getTotalPosition (pos)));
      //
      // GridCoordinateFP2D size = shrinkCoord (yeeLayout.getEpsCoordFP (GammaM.getTotalSize ()));
      //
      // if (posAbs.getX () >= size.getX () / 2 - 20 && posAbs.getX () < size.getX () / 2 + 20
      //     && posAbs.getY () >= 50 && posAbs.getY () < size.getY () - 50)
      // {
      //   valGamma->setCurValue (1);
      // }

      // if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
      //     + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
      // {
      //   valGamma->setCurValue (1);
      // }

      GammaM.setFieldPointValue (valGamma, pos);
    }
  }

  dumper.init (0, CURRENT, processId, "OmegaPE");
  dumper.dumpGrid (OmegaPE);

  dumper.init (0, CURRENT, processId, "OmegaPM");
  dumper.dumpGrid (OmegaPM);

  dumper.init (0, CURRENT, processId, "GammaE");
  dumper.dumpGrid (GammaE);

  dumper.init (0, CURRENT, processId, "GammaM");
  dumper.dumpGrid (GammaM);

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
    dumper.init (0, CURRENT, processId, "SigmaX");
    dumper.dumpGrid (SigmaX);
  }

  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (0, CURRENT, processId, "SigmaY");
    dumper.dumpGrid (SigmaY);
  }

  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (0, CURRENT, processId, "SigmaZ");
    dumper.dumpGrid (SigmaZ);
  }

  for (int i = 0; i < Ez.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ez.getSize ().getY (); ++j)
    {
      FieldPointValue* valEz = new FieldPointValue ();

      FieldPointValue* valDz = new FieldPointValue ();

      FieldPointValue* valD1z = new FieldPointValue ();

      FieldPointValue* valEzAmp;
      if (calculateAmplitude)
      {
        valEzAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Ez.setFieldPointValue (valEz, pos);

      Dz.setFieldPointValue (valDz, pos);

      D1z.setFieldPointValue (valD1z, pos);

      if (calculateAmplitude)
      {
        EzAmplitude.setFieldPointValue (valEzAmp, pos);
      }
    }
  }

  for (int i = 0; i < Hx.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hx.getSize ().getY (); ++j)
    {
      FieldPointValue* valHx = new FieldPointValue ();

      FieldPointValue* valBx = new FieldPointValue ();

      FieldPointValue* valB1x = new FieldPointValue ();

      FieldPointValue* valHxAmp;
      if (calculateAmplitude)
      {
        valHxAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Hx.setFieldPointValue (valHx, pos);

      Bx.setFieldPointValue (valBx, pos);

      B1x.setFieldPointValue (valB1x, pos);

      if (calculateAmplitude)
      {
        HxAmplitude.setFieldPointValue (valHxAmp, pos);
      }
    }
  }

  for (int i = 0; i < Hy.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hy.getSize ().getY (); ++j)
    {
      FieldPointValue* valHy = new FieldPointValue ();

      FieldPointValue* valBy = new FieldPointValue ();

      FieldPointValue* valB1y = new FieldPointValue ();

      FieldPointValue* valHyAmp;
      if (calculateAmplitude)
      {
        valHyAmp = new FieldPointValue ();
      }

      GridCoordinate2D pos (i, j);

      Hy.setFieldPointValue (valHy, pos);

      By.setFieldPointValue (valBy, pos);

      B1y.setFieldPointValue (valB1y, pos);

      if (calculateAmplitude)
      {
        HyAmplitude.setFieldPointValue (valHyAmp, pos);
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

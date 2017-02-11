#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "Kernels.h"
#include "Scheme3D.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_3D

void
Scheme3D::performPlaneWaveESteps (time_step t)
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
    if (incidentWaveAngle2 == PhysicsConst::Pi / 4)
    {
      relPhi = sqrt (2) * asin (sin(arg) / (S * sqrt (2))) / asin (sin (arg) / S);
    }
    else
    {
      ASSERT (incidentWaveAngle2 == 0);

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

  /*
   * FIXME: add assert that right border is reached
   */

  EInc.nextTimeStep ();
}

void
Scheme3D::performPlaneWaveHSteps (time_step t)
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
    if (incidentWaveAngle2 == PhysicsConst::Pi / 4)
    {
      relPhi = sqrt (2) * asin (sin(arg) / (S * sqrt (2))) / asin (sin (arg) / S);
    }
    else
    {
      ASSERT (incidentWaveAngle2 == 0);

      relPhi = 1;
    }

    FieldValue val = valH->getPrevValue () + (gridTimeStep / (relPhi * PhysicsConst::Mu0 * gridStep)) * (valE1->getPrevValue () - valE2->getPrevValue ());

    valH->setCurValue (val);
  }

  HInc.nextTimeStep ();
}

void
Scheme3D::performExSteps (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
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
Scheme3D::calculateExStep (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ex.getTotalPosition (pos);

        GridCoordinate3D posDown = yeeLayout.getExCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getExCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout.getExCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getExCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEx = Ex.getFieldPointValue (pos);

        FieldPointValue* valEps = Eps.getFieldPointValue (pos);

        FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
        FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

        FieldPointValue* valHy1 = Hy.getFieldPointValue (posFront);
        FieldPointValue* valHy2 = Hy.getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz1 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz2 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy1 -= diff * (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy2 -= diff * (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = valEps->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = valEps->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FieldValue val = calculateEx_3D (valEx->getPrevValue (),
                                         prevHz1,
                                         prevHz2,
                                         prevHy1,
                                         prevHy2,
                                         gridTimeStep,
                                         gridStep,
                                         eps * eps0);

        valEx->setCurValue (val);
      }
    }
  }
}

void
Scheme3D::calculateExStepPML (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ex.getTotalPosition (pos);

        GridCoordinate3D posDown = yeeLayout.getExCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getExCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout.getExCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getExCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valDx = Dx.getFieldPointValue (pos);

        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

        FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
        FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

        FieldPointValue* valHy1 = Hy.getFieldPointValue (posFront);
        FieldPointValue* valHy2 = Hy.getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz1 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz2 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy1 -= diff * (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy2 -= diff * (- sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

        /*
         * FIXME: precalculate coefficients
         */
        FPValue k_y = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
        FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

        FieldValue val = calculateEx_3D_Precalc (valDx->getPrevValue (),
                                                 prevHz1,
                                                 prevHz2,
                                                 prevHy1,
                                                 prevHy2,
                                                 Ca,
                                                 Cb);

        valDx->setCurValue (val);
      }
    }
  }

  if (useMetamaterials)
  {
    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          GridCoordinate3D posAbs = Ex.getTotalPosition (pos);

          GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (posAbs);

          FieldPointValue* valD1x = D1x.getFieldPointValue (pos);
          FieldPointValue* valDx = Dx.getFieldPointValue (pos);

          FieldPointValue* valOmegaPE1 = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0))));
          FieldPointValue* valOmegaPE2 = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0))));
          FieldPointValue* valGammaE1 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0))));
          FieldPointValue* valGammaE2 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0))));

          FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0))));
          FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0))));

#ifdef COMPLEX_FIELD_VALUES
          FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue ().real () + valOmegaPE2->getCurValue ().real ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue ().real () + valGammaE2->getCurValue ().real ()) / 2;
#else /* COMPLEX_FIELD_VALUES */
          FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue () + valOmegaPE2->getCurValue ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue () + valGammaE2->getCurValue ()) / 2;
#endif /* !COMPLEX_FIELD_VALUES */

          /*
           * FIXME: precalculate coefficients
           */
          FPValue A = 4*eps0*eps + 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE;

          FieldValue val = calculateDrudeE (valDx->getCurValue (),
                                            valDx->getPrevValue (),
                                            valDx->getPrevPrevValue (),
                                            valD1x->getPrevValue (),
                                            valD1x->getPrevPrevValue (),
                                            (4 + 2*gridTimeStep*gammaE) / A,
                                            -8 / A,
                                            (4 - 2*gridTimeStep*gammaE) / A,
                                            (2*eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE - 8*eps0*eps) / A,
                                            (4*eps0*eps - 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE) / A);

          valD1x->setCurValue (val);
        }
      }
    }
  }

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ex.getTotalPosition (pos);

        FieldPointValue* valEx = Ex.getFieldPointValue (pos);

        FieldPointValue* valDx;

        if (useMetamaterials)
        {
          valDx = D1x.getFieldPointValue (pos);
        }
        else
        {
          valDx = Dx.getFieldPointValue (pos);
        }

        FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

        GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (posAbs);
        FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0))));
        FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0.5, 0, 0))));

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

        FPValue sigmaX = valSigmaX->getCurValue ().real ();
        FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

        FPValue sigmaX = valSigmaX->getCurValue ();
        FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue modifier = eps * eps0;
        if (useMetamaterials)
        {
          modifier = 1;
        }

        FPValue k_x = 1;
        FPValue k_z = 1;

        FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
        FPValue Cb = ((2 * eps0 * k_x + sigmaX * gridTimeStep) / (modifier)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
        FPValue Cc = ((2 * eps0 * k_x - sigmaX * gridTimeStep) / (modifier)) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

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
}

void
Scheme3D::performEySteps (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
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
Scheme3D::calculateEyStep (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ey.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getEyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getEyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout.getEyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getEyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEy = Ey.getFieldPointValue (pos);

        FieldPointValue* valEps = Eps.getFieldPointValue (pos);

        FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
        FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx.getFieldPointValue (posFront);
        FieldPointValue* valHx2 = Hx.getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz1 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz2 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx1 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx2 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = valEps->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = valEps->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FieldValue val = calculateEy_3D (valEy->getPrevValue (),
                                         prevHx1,
                                         prevHx2,
                                         prevHz1,
                                         prevHz2,
                                         gridTimeStep,
                                         gridStep,
                                         eps * eps0);

        valEy->setCurValue (val);
      }
    }
  }
}

void
Scheme3D::calculateEyStepPML (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ey.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getEyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getEyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout.getEyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getEyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valDy = Dy.getFieldPointValue (pos);

        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

        FieldPointValue* valHz1 = Hz.getFieldPointValue (posRight);
        FieldPointValue* valHz2 = Hz.getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx.getFieldPointValue (posFront);
        FieldPointValue* valHx2 = Hx.getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz1 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHz2 -= -diff * cos (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx1 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx2 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

        /*
         * FIXME: precalculate coefficients
         */
        FPValue k_z = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
        FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

        FieldValue val = calculateEy_3D_Precalc (valDy->getPrevValue (),
                                                 prevHx1,
                                                 prevHx2,
                                                 prevHz1,
                                                 prevHz2,
                                                 Ca,
                                                 Cb);

        valDy->setCurValue (val);
      }
    }
  }

  if (useMetamaterials)
  {
    for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
    {
      for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
      {
        for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          GridCoordinate3D posAbs = Ey.getTotalPosition (pos);

          GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (posAbs);

          FieldPointValue* valD1y = D1y.getFieldPointValue (pos);
          FieldPointValue* valDy = Dy.getFieldPointValue (pos);

          FieldPointValue* valOmegaPE1 = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0))));
          FieldPointValue* valOmegaPE2 = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0))));
          FieldPointValue* valGammaE1 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0))));
          FieldPointValue* valGammaE2 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0))));

          FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0))));
          FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0))));

  #ifdef COMPLEX_FIELD_VALUES
          FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue ().real () + valOmegaPE2->getCurValue ().real ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue ().real () + valGammaE2->getCurValue ().real ()) / 2;
  #else /* COMPLEX_FIELD_VALUES */
          FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue () + valOmegaPE2->getCurValue ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue () + valGammaE2->getCurValue ()) / 2;
  #endif /* !COMPLEX_FIELD_VALUES */

          /*
           * FIXME: precalculate coefficients
           */
          FPValue A = 4*eps0*eps + 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE;

          FieldValue val = calculateDrudeE (valDy->getCurValue (),
                                            valDy->getPrevValue (),
                                            valDy->getPrevPrevValue (),
                                            valD1y->getPrevValue (),
                                            valD1y->getPrevPrevValue (),
                                            (4 + 2*gridTimeStep*gammaE) / A,
                                            -8 / A,
                                            (4 - 2*gridTimeStep*gammaE) / A,
                                            (2*eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE - 8*eps0*eps) / A,
                                            (4*eps0*eps - 2*gridTimeStep*eps0*eps*gammaE + eps0*gridTimeStep*gridTimeStep*omegaPE*omegaPE) / A);

          valD1y->setCurValue (val);
        }
      }
    }
  }

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ey.getTotalPosition (pos);

        FieldPointValue* valEy = Ey.getFieldPointValue (pos);

        FieldPointValue* valDy;

        if (useMetamaterials)
        {
          valDy = D1y.getFieldPointValue (pos);
        }
        else
        {
          valDy = Dy.getFieldPointValue (pos);
        }

        FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

        GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (posAbs);
        FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0))));
        FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0.5, 0))));

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

        FPValue sigmaX = valSigmaX->getCurValue ().real ();
        FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

        FPValue sigmaX = valSigmaX->getCurValue ();
        FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue modifier = eps * eps0;
        if (useMetamaterials)
        {
          modifier = 1;
        }

        FPValue k_x = 1;
        FPValue k_y = 1;

        FPValue Ca = (2 * eps0 * k_x - sigmaX * gridTimeStep) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
        FPValue Cb = ((2 * eps0 * k_y + sigmaY * gridTimeStep) / (modifier)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);
        FPValue Cc = ((2 * eps0 * k_y - sigmaY * gridTimeStep) / (modifier)) / (2 * eps0 * k_x + sigmaX * gridTimeStep);

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
}

void
Scheme3D::performEzSteps (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
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
Scheme3D::calculateEzStep (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ez.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getEzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getEzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout.getEzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getEzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valEz = Ez.getFieldPointValue (pos);

        FieldPointValue* valEps = Eps.getFieldPointValue (pos);

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
          if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT, true))
          {
            /*
             * HInc: 0, 1, etc.
             */
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy1 -= diff * ( - sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy2 -= diff * ( - sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }

          if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx1 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx2 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue eps = valEps->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue eps = valEps->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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
}

void
Scheme3D::calculateEzStepPML (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Ez.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getEzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getEzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout.getEzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getEzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valDz = Dz.getFieldPointValue (pos);

        FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);

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
          if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT, true))
          {
            /*
             * HInc: 0, 1, etc.
             */
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy1 -= diff * ( - sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHy2 -= diff * ( - sin (incidentWaveAngle3) * cos (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }

          if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx1 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1) - 0.5;
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valH1 = HInc.getFieldPointValue (pos1);
            FieldPointValue *valH2 = HInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valH1->getPrevValue () + proportionD2 * valH2->getPrevValue ();

            prevHx2 -= diff * (sin (incidentWaveAngle3) * sin (incidentWaveAngle2) + cos (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

        /*
         * FIXME: precalculate coefficients
         */
        FPValue k_x = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaX = valSigmaX->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaX = valSigmaX->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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
  }

  if (useMetamaterials)
  {
    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Ez.getTotalPosition (pos);

          FieldPointValue* valEz = Ez.getFieldPointValue (pos);

          GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (posAbs);

          FieldPointValue* valD1z = D1z.getFieldPointValue (pos);
          FieldPointValue* valDz = Dz.getFieldPointValue (pos);

          GridCoordinateFP3D coord1 = realCoord + GridCoordinateFP3D (0, 0, 0.5);
          GridCoordinate3D coord2 = yeeLayout.getEpsCoord (coord1);
          GridCoordinate3D coord3 = OmegaPE.getRelativePosition (coord2);

          FieldPointValue* valOmegaPE1 = OmegaPE.getFieldPointValue (coord3);
          FieldPointValue* valOmegaPE2 = OmegaPE.getFieldPointValue (OmegaPE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5))));
          FieldPointValue* valGammaE1 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5))));
          FieldPointValue* valGammaE2 = GammaE.getFieldPointValue (GammaE.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5))));

          FieldPointValue* valEps1 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0, 0.5))));
          FieldPointValue* valEps2 = Eps.getFieldPointValue (Eps.getRelativePosition (yeeLayout.getEpsCoord (realCoord - GridCoordinateFP3D (0, 0, 0.5))));

  #ifdef COMPLEX_FIELD_VALUES
          FPValue eps = (valEps1->getCurValue ().real () + valEps2->getCurValue ().real ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue ().real () + valOmegaPE2->getCurValue ().real ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue ().real () + valGammaE2->getCurValue ().real ()) / 2;
  #else /* COMPLEX_FIELD_VALUES */
          FPValue eps = (valEps1->getCurValue () + valEps2->getCurValue ()) / 2;

          FPValue omegaPE = (valOmegaPE1->getCurValue () + valOmegaPE2->getCurValue ()) / 2;
          FPValue gammaE = (valGammaE1->getCurValue () + valGammaE2->getCurValue ()) / 2;
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
  }

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

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

        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);
        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);
        FieldPointValue* valEps = Eps.getFieldPointValue (pos);

        FPValue k_y = 1;
        FPValue k_z = 1;

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
}

void
Scheme3D::performHxSteps (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
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
Scheme3D::calculateHxStep (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hx.getTotalPosition (pos);

        GridCoordinate3D posDown = yeeLayout.getHxCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getHxCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout.getHxCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getHxCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHx = Hx.getFieldPointValue (pos);

        FieldPointValue* valMu = Mu.getFieldPointValue (pos);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (posUp);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (posDown);

        FieldPointValue* valEy1 = Ey.getFieldPointValue (posFront);
        FieldPointValue* valEy2 = Ey.getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz2 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz1 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy2 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy1 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FieldValue val = calculateHx_3D (valHx->getPrevValue (),
                                         prevEy1,
                                         prevEy2,
                                         prevEz1,
                                         prevEz2,
                                         gridTimeStep,
                                         gridStep,
                                         mu * mu0);

        valHx->setCurValue (val);
      }
    }
  }
}

void
Scheme3D::calculateHxStepPML (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hx.getTotalPosition (pos);

        GridCoordinate3D posDown = yeeLayout.getHxCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getHxCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout.getHxCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getHxCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valBx = Bx.getFieldPointValue (pos);

        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (posUp);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (posDown);

        FieldPointValue* valEy1 = Ey.getFieldPointValue (posFront);
        FieldPointValue* valEy2 = Ey.getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz2 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz1 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy2 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy1 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

        FPValue k_y = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
        FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

        FieldValue val = calculateHx_3D_Precalc (valBx->getPrevValue (),
                                                 prevEy1,
                                                 prevEy2,
                                                 prevEz1,
                                                 prevEz2,
                                                 Ca,
                                                 Cb);

        valBx->setCurValue (val);
      }
    }
  }

  if (useMetamaterials)
  {
    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hx.getTotalPosition (pos);

          FieldPointValue* valHx = Hx.getFieldPointValue (pos);
          GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (posAbs);

          FieldPointValue* valB1x = B1x.getFieldPointValue (pos);
          FieldPointValue* valBx = Bx.getFieldPointValue (pos);

          FieldPointValue* valOmegaPM1 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0.5))));
          FieldPointValue* valOmegaPM2 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, 0.5))));
          FieldPointValue* valOmegaPM3 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, -0.5))));
          FieldPointValue* valOmegaPM4 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, -0.5))));

          FieldPointValue* valGammaM1 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0.5))));
          FieldPointValue* valGammaM2 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, 0.5))));
          FieldPointValue* valGammaM3 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, -0.5))));
          FieldPointValue* valGammaM4 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, -0.5))));

          FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, 0.5))));
          FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, 0.5))));
          FieldPointValue* valMu3 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, 0.5, -0.5))));
          FieldPointValue* valMu4 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0, -0.5, -0.5))));

          FPValue omegaPM;
          FPValue gammaM;
          FPValue dividerOmega = 0;
          FPValue dividerGamma = 0;

#ifdef COMPLEX_FIELD_VALUES
          FPValue mu1 = valMu1->getCurValue ().real ();
          FPValue mu2 = valMu2->getCurValue ().real ();
          FPValue mu3 = valMu3->getCurValue ().real ();
          FPValue mu4 = valMu4->getCurValue ().real ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ().real ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ().real ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ().real ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ().real ();

          FPValue gammaM1 = valGammaM1->getCurValue ().real ();
          FPValue gammaM2 = valGammaM2->getCurValue ().real ();
          FPValue gammaM3 = valGammaM3->getCurValue ().real ();
          FPValue gammaM4 = valGammaM4->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
          FPValue mu1 = valMu1->getCurValue ();
          FPValue mu2 = valMu2->getCurValue ();
          FPValue mu3 = valMu3->getCurValue ();
          FPValue mu4 = valMu4->getCurValue ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ();

          FPValue gammaM1 = valGammaM1->getCurValue ();
          FPValue gammaM2 = valGammaM2->getCurValue ();
          FPValue gammaM3 = valGammaM3->getCurValue ();
          FPValue gammaM4 = valGammaM4->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          // if (valOmegaPM1->getCurValue () == 0
          //     || valOmegaPM2->getCurValue () == 0)
          // {
          //   dividerOmega = sqrtf (2.0);
          //   dividerGamma = 2.0;
          // }
          // else
          // {
          //   if (valOmegaPM1->getCurValue () != valOmegaPM2->getCurValue ()
          //       || valGammaM1->getCurValue () != valGammaM2->getCurValue ())
          //   {
          //     ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
          //   }
          //
          //   dividerOmega = 2.0;
          //   dividerGamma = 2.0;
          // }
          //
          // ASSERT (dividerOmega != 0);
          // ASSERT (dividerGamma != 0);
          dividerOmega = 4.0;
          dividerGamma = 4.0;

          omegaPM = (omegaPM1 + omegaPM2 + omegaPM3 + omegaPM4) / dividerOmega;
          gammaM = (gammaM1 + gammaM2 + gammaM3 + gammaM4) / dividerGamma;
          FPValue mu = (mu1 + mu2 + mu3 + mu4) / 4;

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
  }

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

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

        FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

        FieldPointValue* valMu = Mu.getFieldPointValue (pos);

        FPValue k_x = 1;
        FPValue k_z = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();

        FPValue sigmaX = valSigmaX->getCurValue ().real ();
        FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();

        FPValue sigmaX = valSigmaX->getCurValue ();
        FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue modifier = mu * mu0;
        if (useMetamaterials)
        {
          modifier = 1;
        }

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
}

void
Scheme3D::performHySteps (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
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
Scheme3D::calculateHyStep (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hy.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getHyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getHyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout.getHyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getHyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHy = Hy.getFieldPointValue (pos);

        FieldPointValue* valMu = Mu.getFieldPointValue (pos);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (posRight);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex.getFieldPointValue (posFront);
        FieldPointValue* valEx2 = Ex.getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz2 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz1 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx2 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx1 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FieldValue val = calculateHy_3D (valHy->getPrevValue (),
                                         prevEz1,
                                         prevEz2,
                                         prevEx1,
                                         prevEx2,
                                         gridTimeStep,
                                         gridStep,
                                         mu * mu0);

        valHy->setCurValue (val);
      }
    }
  }
}

void
Scheme3D::calculateHyStepPML (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hy.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getHyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getHyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout.getHyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout.getHyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valBy = By.getFieldPointValue (pos);

        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (posRight);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex.getFieldPointValue (posFront);
        FieldPointValue* valEx2 = Ex.getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        if (useTFSF)
        {
          if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz2 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }
          else if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEz1 += diff * sin (incidentWaveAngle3) * sin (incidentWaveAngle1);
          }

          if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::BACK, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posBack));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx2 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::FRONT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posFront));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx1 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
        }

        FPValue k_z = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue Ca = (2 * eps0 * k_z - sigmaZ * gridTimeStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);
        FPValue Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_z + sigmaZ * gridTimeStep);

        FieldValue val = calculateHy_3D_Precalc (valBy->getPrevValue (),
                                                 prevEz1,
                                                 prevEz2,
                                                 prevEx1,
                                                 prevEx2,
                                                 Ca,
                                                 Cb);

        valBy->setCurValue (val);
      }
    }
  }

  if (useMetamaterials)
  {
    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hy.getTotalPosition (pos);

          FieldPointValue* valHy = Hy.getFieldPointValue (pos);
          GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (posAbs);

          FieldPointValue* valB1y = B1y.getFieldPointValue (pos);
          FieldPointValue* valBy = By.getFieldPointValue (pos);

          FieldPointValue* valOmegaPM1 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0.5))));
          FieldPointValue* valOmegaPM2 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, -0.5))));
          FieldPointValue* valOmegaPM3 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, 0.5))));
          FieldPointValue* valOmegaPM4 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, -0.5))));

          FieldPointValue* valGammaM1 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0.5))));
          FieldPointValue* valGammaM2 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, -0.5))));
          FieldPointValue* valGammaM3 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, 0.5))));
          FieldPointValue* valGammaM4 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, -0.5))));

          FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, 0.5))));
          FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0, -0.5))));
          FieldPointValue* valMu3 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, 0.5))));
          FieldPointValue* valMu4 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0, -0.5))));

          FPValue omegaPM;
          FPValue gammaM;
          FPValue dividerOmega = 0;
          FPValue dividerGamma = 0;

#ifdef COMPLEX_FIELD_VALUES
          FPValue mu1 = valMu1->getCurValue ().real ();
          FPValue mu2 = valMu2->getCurValue ().real ();
          FPValue mu3 = valMu3->getCurValue ().real ();
          FPValue mu4 = valMu4->getCurValue ().real ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ().real ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ().real ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ().real ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ().real ();

          FPValue gammaM1 = valGammaM1->getCurValue ().real ();
          FPValue gammaM2 = valGammaM2->getCurValue ().real ();
          FPValue gammaM3 = valGammaM3->getCurValue ().real ();
          FPValue gammaM4 = valGammaM4->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
          FPValue mu1 = valMu1->getCurValue ();
          FPValue mu2 = valMu2->getCurValue ();
          FPValue mu3 = valMu3->getCurValue ();
          FPValue mu4 = valMu4->getCurValue ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ();

          FPValue gammaM1 = valGammaM1->getCurValue ();
          FPValue gammaM2 = valGammaM2->getCurValue ();
          FPValue gammaM3 = valGammaM3->getCurValue ();
          FPValue gammaM4 = valGammaM4->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          // if (valOmegaPM1->getCurValue () == 0
          //     || valOmegaPM2->getCurValue () == 0)
          // {
          //   dividerOmega = sqrtf (2.0);
          //   dividerGamma = 2.0;
          // }
          // else
          // {
          //   if (valOmegaPM1->getCurValue () != valOmegaPM2->getCurValue ()
          //       || valGammaM1->getCurValue () != valGammaM2->getCurValue ())
          //   {
          //     ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
          //   }
          //
          //   dividerOmega = 2.0;
          //   dividerGamma = 2.0;
          // }
          //
          // ASSERT (dividerOmega != 0);
          // ASSERT (dividerGamma != 0);
          dividerOmega = 4.0;
          dividerGamma = 4.0;

          omegaPM = (omegaPM1 + omegaPM2 + omegaPM3 + omegaPM4) / dividerOmega;
          gammaM = (gammaM1 + gammaM2 + gammaM3 + gammaM4) / dividerGamma;
          FPValue mu = (mu1 + mu2 + mu3 + mu4) / 4;

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
  }

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

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

        FieldPointValue* valSigmaX = SigmaX.getFieldPointValue (pos);
        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);

        FieldPointValue* valMu = Mu.getFieldPointValue (pos);

        FPValue k_x = 1;
        FPValue k_y = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();

        FPValue sigmaX = valSigmaX->getCurValue ().real ();
        FPValue sigmaY = valSigmaY->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();

        FPValue sigmaX = valSigmaX->getCurValue ();
        FPValue sigmaY = valSigmaY->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue modifier = mu * mu0;
        if (useMetamaterials)
        {
          modifier = 1;
        }

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
}

void
Scheme3D::performHzSteps (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
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
Scheme3D::calculateHzStep (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hz.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getHzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getHzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout.getHzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getHzCircuitElement (pos, LayoutDirection::UP);

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
          if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx2 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx1 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }

          if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy2 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy1 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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
}

void
Scheme3D::calculateHzStepPML (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hz.getTotalPosition (pos);

        GridCoordinate3D posLeft = yeeLayout.getHzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout.getHzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout.getHzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout.getHzCircuitElement (pos, LayoutDirection::UP);

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
          if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posDown));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx2 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (posUp));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEx1 += diff * (cos (incidentWaveAngle3) * sin (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * cos (incidentWaveAngle2));
          }

          if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posLeft));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy2 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
          else if (yeeLayout.doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT, true))
          {
            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (posRight));
            GridCoordinateFP3D zeroCoordFP = yeeLayout.getZeroIncCoordFP ();

            FPValue x = realCoord.getX () - zeroCoordFP.getX ();
            FPValue y = realCoord.getY () - zeroCoordFP.getY ();
            FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
            FPValue d = x * sin (incidentWaveAngle1) * cos (incidentWaveAngle2)
                           + y * sin (incidentWaveAngle1) * sin (incidentWaveAngle2)
                           + z * cos (incidentWaveAngle1);
            FPValue coordD1 = (FPValue) ((grid_iter) d);
            FPValue coordD2 = coordD1 + 1;
            FPValue proportionD2 = d - coordD1;
            FPValue proportionD1 = 1 - proportionD2;

            GridCoordinate1D pos1 (coordD1);
            GridCoordinate1D pos2 (coordD2);

            FieldPointValue *valE1 = EInc.getFieldPointValue (pos1);
            FieldPointValue *valE2 = EInc.getFieldPointValue (pos2);

            FieldValue diff = proportionD1 * valE1->getPrevValue () + proportionD2 * valE2->getPrevValue ();

            prevEy1 += diff * ( - cos (incidentWaveAngle3) * cos (incidentWaveAngle2) - sin (incidentWaveAngle3) * cos (incidentWaveAngle1) * sin (incidentWaveAngle2));
          }
        }

        FPValue k_x = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue sigmaX = valSigmaX->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue sigmaX = valSigmaX->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

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
  }

  if (useMetamaterials)
  {
    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          GridCoordinate3D posAbs = Hz.getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (posAbs);

          FieldPointValue* valB1z = B1z.getFieldPointValue (pos);
          FieldPointValue* valBz = Bz.getFieldPointValue (pos);

          FieldPointValue* valOmegaPM1 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0.5, 0))));
          FieldPointValue* valOmegaPM2 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, -0.5, 0))));
          FieldPointValue* valOmegaPM3 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0.5, 0))));
          FieldPointValue* valOmegaPM4 = OmegaPM.getFieldPointValue (OmegaPM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, -0.5, 0))));

          FieldPointValue* valGammaM1 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0.5, 0))));
          FieldPointValue* valGammaM2 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, -0.5, 0))));
          FieldPointValue* valGammaM3 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, 0.5, 0))));
          FieldPointValue* valGammaM4 = GammaM.getFieldPointValue (GammaM.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (-0.5, -0.5, 0))));

          FieldPointValue* valMu1 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0.5, 0))));
          FieldPointValue* valMu2 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, -0.5, 0))));
          FieldPointValue* valMu3 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, 0.5, 0))));
          FieldPointValue* valMu4 = Mu.getFieldPointValue (Mu.getRelativePosition (yeeLayout.getEpsCoord (realCoord + GridCoordinateFP3D (0.5, -0.5, 0))));

          FPValue omegaPM;
          FPValue gammaM;
          FPValue dividerOmega = 0;
          FPValue dividerGamma = 0;

#ifdef COMPLEX_FIELD_VALUES
          FPValue mu1 = valMu1->getCurValue ().real ();
          FPValue mu2 = valMu2->getCurValue ().real ();
          FPValue mu3 = valMu3->getCurValue ().real ();
          FPValue mu4 = valMu4->getCurValue ().real ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ().real ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ().real ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ().real ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ().real ();

          FPValue gammaM1 = valGammaM1->getCurValue ().real ();
          FPValue gammaM2 = valGammaM2->getCurValue ().real ();
          FPValue gammaM3 = valGammaM3->getCurValue ().real ();
          FPValue gammaM4 = valGammaM4->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
          FPValue mu1 = valMu1->getCurValue ();
          FPValue mu2 = valMu2->getCurValue ();
          FPValue mu3 = valMu3->getCurValue ();
          FPValue mu4 = valMu4->getCurValue ();

          FPValue omegaPM1 = valOmegaPM1->getCurValue ();
          FPValue omegaPM2 = valOmegaPM2->getCurValue ();
          FPValue omegaPM3 = valOmegaPM3->getCurValue ();
          FPValue omegaPM4 = valOmegaPM4->getCurValue ();

          FPValue gammaM1 = valGammaM1->getCurValue ();
          FPValue gammaM2 = valGammaM2->getCurValue ();
          FPValue gammaM3 = valGammaM3->getCurValue ();
          FPValue gammaM4 = valGammaM4->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          // if (valOmegaPM1->getCurValue () == 0
          //     || valOmegaPM2->getCurValue () == 0)
          // {
          //   dividerOmega = sqrtf (2.0);
          //   dividerGamma = 2.0;
          // }
          // else
          // {
          //   if (valOmegaPM1->getCurValue () != valOmegaPM2->getCurValue ()
          //       || valGammaM1->getCurValue () != valGammaM2->getCurValue ())
          //   {
          //     ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
          //   }
          //
          //   dividerOmega = 2.0;
          //   dividerGamma = 2.0;
          // }
          //
          // ASSERT (dividerOmega != 0);
          // ASSERT (dividerGamma != 0);
          dividerOmega = 4.0;
          dividerGamma = 4.0;

          omegaPM = (omegaPM1 + omegaPM2 + omegaPM3 + omegaPM4) / dividerOmega;
          gammaM = (gammaM1 + gammaM2 + gammaM3 + gammaM4) / dividerGamma;
          FPValue mu = (mu1 + mu2 + mu3 + mu4) / 4;

          /*
           * FIXME: precalculate coefficients
           */
          FPValue C = 4*mu0*mu + 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM;

          FieldValue val = calculateDrudeH (valBz->getCurValue (),
                                            valBz->getPrevValue (),
                                            valBz->getPrevPrevValue (),
                                            valB1z->getPrevValue (),
                                            valB1z->getPrevPrevValue (),
                                            (4 + 2*gridTimeStep*gammaM) / C,
                                            -8 / C,
                                            (4 - 2*gridTimeStep*gammaM) / C,
                                            (2*mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM - 8*mu0*mu) / C,
                                            (4*mu0*mu - 2*gridTimeStep*mu0*mu*gammaM + mu0*gridTimeStep*gridTimeStep*omegaPM*omegaPM) / C);

          valB1z->setCurValue (val);
        }
      }
    }
  }

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        GridCoordinate3D posAbs = Hz.getTotalPosition (pos);

        FieldPointValue* valHz = Hz.getFieldPointValue (pos);

        FieldPointValue* valBz;

        if (useMetamaterials)
        {
          valBz = B1z.getFieldPointValue (pos);
        }
        else
        {
          valBz = Bz.getFieldPointValue (pos);
        }

        FieldPointValue* valSigmaY = SigmaY.getFieldPointValue (pos);
        FieldPointValue* valSigmaZ = SigmaZ.getFieldPointValue (pos);

        // GridCoordinateFP2D realCoord = shrinkCoord (yeeLayout.getHzCoordFP (posAbs));
        // FieldPointValue* valMu1 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
        // FieldPointValue* valMu2 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () + 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
        // FieldPointValue* valMu3 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () + 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
        // FieldPointValue* valMu4 = Mu.getFieldPointValue (yeeLayout.getMuCoord (GridCoordinateFP3D (realCoord.getX () - 0.5, realCoord.getY () - 0.5, yeeLayout.getMinMuCoordFP ().getZ ())));
        // FieldValue mu = (valMu1->getCurValue () + valMu2->getCurValue () + valMu3->getCurValue () + valMu4->getCurValue ()) / 4;
        FieldPointValue* valMu = Mu.getFieldPointValue (pos);

        FPValue k_y = 1;
        FPValue k_z = 1;

#ifdef COMPLEX_FIELD_VALUES
        FPValue mu = valMu->getCurValue ().real ();

        FPValue sigmaY = valSigmaY->getCurValue ().real ();
        FPValue sigmaZ = valSigmaZ->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
        FPValue mu = valMu->getCurValue ();

        FPValue sigmaY = valSigmaY->getCurValue ();
        FPValue sigmaZ = valSigmaZ->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        FPValue modifier = mu * mu0;
        if (useMetamaterials)
        {
          modifier = 1;
        }

        FPValue Ca = (2 * eps0 * k_y - sigmaY * gridTimeStep) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
        FPValue Cb = ((2 * eps0 * k_z + sigmaZ * gridTimeStep) / (modifier)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);
        FPValue Cc = ((2 * eps0 * k_z - sigmaZ * gridTimeStep) / (modifier)) / (2 * eps0 * k_y + sigmaY * gridTimeStep);

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
}

void
Scheme3D::performNSteps (time_step startStep, time_step numberTimeSteps, int dumpRes)
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  GridCoordinate3D EzSize = Ez.getSize ();

  time_step stepLimit = startStep + numberTimeSteps;

  for (int t = startStep; t < stepLimit; ++t)
  {
    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getComputationStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getComputationEnd ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getComputationStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getComputationEnd ());

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getComputationStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getComputationEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getComputationStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getComputationEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getComputationStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getComputationEnd ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getComputationStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getComputationEnd ());

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);
    performEzSteps (t, EzStart, EzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      //if (processId == 0)
#endif
      {
        grid_coord start;
        grid_coord end;
#ifdef PARALLEL_GRID
        start = processId == 0 ? yeeLayout.getLeftBorderPML ().getZ () : 0;
        end = processId == ParallelGrid::getParallelCore ()->getTotalProcCount () - 1 ? Ez.getRelativePosition (yeeLayout.getRightBorderPML ()).getZ () : Ez.getCurrentSize ().getZ ();
#else /* PARALLEL_GRID */
        start = yeeLayout.getLeftBorderPML ().getZ ();
        end = yeeLayout.getRightBorderPML ().getZ ();
#endif /* !PARALLEL_GRID */
        for (grid_coord k = start; k < end; ++k)
        {
          GridCoordinate3D pos (EzSize.getX () / 8, EzSize.getY () / 2, k);
          FieldPointValue* tmp = Ez.getFieldPointValue (pos);

  #ifdef COMPLEX_FIELD_VALUES
          tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                        cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
  #else /* COMPLEX_FIELD_VALUES */
          tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
  #endif /* !COMPLEX_FIELD_VALUES */
        }
      }
    }

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();
    Ez.nextTimeStep ();

    if (usePML)
    {
      Dx.nextTimeStep ();
      Dy.nextTimeStep ();
      Dz.nextTimeStep ();
    }

    if (useMetamaterials)
    {
      D1x.nextTimeStep ();
      D1y.nextTimeStep ();
      D1z.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);
    performHzSteps (t, HzStart, HzEnd);

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
    Hz.nextTimeStep ();

    if (usePML)
    {
      Bx.nextTimeStep ();
      By.nextTimeStep ();
      Bz.nextTimeStep ();
    }

    if (useMetamaterials)
    {
      B1x.nextTimeStep ();
      B1y.nextTimeStep ();
      B1z.nextTimeStep ();
    }

    /*
     * FIXME: add dump step
     */
    if (t % 100 == 0)
    {
      if (dumpRes)
      {
        TXTDumper<GridCoordinate3D> dumperEx;
        dumperEx.init (t, CURRENT, processId, "3D-in-time-Ex");
        dumperEx.dumpGrid (Ex);

        TXTDumper<GridCoordinate3D> dumperEy;
        dumperEy.init (t, CURRENT, processId, "3D-in-time-Ey");
        dumperEy.dumpGrid (Ey);

        TXTDumper<GridCoordinate3D> dumperEz;
        dumperEz.init (t, CURRENT, processId, "3D-in-time-Ez");
        dumperEz.dumpGrid (Ez);

        TXTDumper<GridCoordinate3D> dumperHx;
        dumperHx.init (t, CURRENT, processId, "3D-in-time-Hx");
        dumperHx.dumpGrid (Hx);

        TXTDumper<GridCoordinate3D> dumperHy;
        dumperHy.init (t, CURRENT, processId, "3D-in-time-Hy");
        dumperHy.dumpGrid (Hy);

        TXTDumper<GridCoordinate3D> dumperHz;
        dumperHz.init (t, CURRENT, processId, "3D-in-time-Hz");
        dumperHz.dumpGrid (Hz);
        //
        // BMPDumper<GridCoordinate3D> dumperHx;
        // dumperHx.init (t, CURRENT, processId, "2D-TMz-in-time-Hx");
        // dumperHx.dumpGrid (Hx);
        //
        // BMPDumper<GridCoordinate3D> dumperHy;
        // dumperHy.init (t, CURRENT, processId, "2D-TMz-in-time-Hy");
        // dumperHy.dumpGrid (Hy);
      }
    }
  }

  if (dumpRes)
  {
    /*
     * FIXME: leave only one dumper
     */
    BMPDumper<GridCoordinate3D> dumperEx;
    dumperEx.init (stepLimit, CURRENT, processId, "3D-in-time-Ex");
    dumperEx.dumpGrid (Ex);

    BMPDumper<GridCoordinate3D> dumperEy;
    dumperEy.init (stepLimit, CURRENT, processId, "3D-in-time-Ey");
    dumperEy.dumpGrid (Ey);

    BMPDumper<GridCoordinate3D> dumperEz;
    dumperEz.init (stepLimit, CURRENT, processId, "3D-in-time-Ez");
    dumperEz.dumpGrid (Ez);

    BMPDumper<GridCoordinate3D> dumperHx;
    dumperHx.init (stepLimit, CURRENT, processId, "3D-in-time-Hx");
    dumperHx.dumpGrid (Hx);

    BMPDumper<GridCoordinate3D> dumperHy;
    dumperHy.init (stepLimit, CURRENT, processId, "3D-in-time-Hy");
    dumperHy.dumpGrid (Hy);

    BMPDumper<GridCoordinate3D> dumperHz;
    dumperHz.init (stepLimit, CURRENT, processId, "3D-in-time-Hz");
    dumperHz.dumpGrid (Hz);

    // BMPDumper<GridCoordinate1D> dumper;
    // dumper.init (stepLimit, PREVIOUS, processId, "3D-incident-E");
    // dumper.dumpGrid (EInc);
    //
    // dumper.init (stepLimit, PREVIOUS, processId, "3D-incident-H");
    // dumper.dumpGrid (HInc);
  }
}

void
Scheme3D::performAmplitudeSteps (time_step startStep, int dumpRes)
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

  GridCoordinate3D EzSize = Ez.getSize ();

  time_step t = startStep;

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FPValue maxAccuracy = -1;

    //is_stable_state = 1;

    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getComputationStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getComputationEnd ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getComputationStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getComputationEnd ());

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getComputationStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getComputationEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getComputationStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getComputationEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getComputationStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getComputationEnd ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getComputationStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getComputationEnd ());

    if (useTFSF)
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);
    performEzSteps (t, EzStart, EzEnd);

    if (!useTFSF)
    {
#if defined (PARALLEL_GRID)
      if (processId == 0)
#endif
      {
        for (grid_coord k = yeeLayout.getLeftBorderPML ().getZ (); k < yeeLayout.getRightBorderPML ().getZ (); ++k)
        {
          GridCoordinate3D pos (EzSize.getX () / 8, EzSize.getY () / 2, k);
          FieldPointValue* tmp = Ez.getFieldPointValue (pos);

  #ifdef COMPLEX_FIELD_VALUES
          tmp->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                        cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
  #else /* COMPLEX_FIELD_VALUES */
          tmp->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
  #endif /* !COMPLEX_FIELD_VALUES */
        }
      }
    }

    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isExInPML (Ex.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ex.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = ExAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getExCoordFP (Ex.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
    {
      for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
      {
        for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isEyInPML (Ey.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ey.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = EyAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getEyCoordFP (Ey.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isEzInPML (Ez.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ez.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = EzAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getEzCoordFP (Ez.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();
    Ez.nextTimeStep ();

    if (usePML)
    {
      Dx.nextTimeStep ();
      Dy.nextTimeStep ();
      Dz.nextTimeStep ();
    }

    if (useTFSF)
    {
      performPlaneWaveHSteps (t);
    }

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);
    performHzSteps (t, HzStart, HzEnd);

    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isHxInPML (Hx.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hx.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HxAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getHxCoordFP (Hx.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isHyInPML (Hy.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hy.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HyAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getHyCoordFP (Hy.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout.isHzInPML (Hz.getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hz.getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HzAmplitude.getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout.getHzCoordFP (Hz.getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout.getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
    Hz.nextTimeStep ();

    if (usePML)
    {
      Bx.nextTimeStep ();
      By.nextTimeStep ();
      Bz.nextTimeStep ();
    }

    ++t;

    if (maxAccuracy < 0)
    {
      is_stable_state = 0;
    }

#if PRINT_MESSAGE
    printf ("%d amplitude calculation step: max accuracy %f. \n", t, maxAccuracy);
#endif /* PRINT_MESSAGE */

    /*
     * FIXME: add dump step
     */
    if (t % 100 == 0)
    {
      if (dumpRes)
      {
        BMPDumper<GridCoordinate3D> dumperEz;
        dumperEz.init (t, CURRENT, processId, "2D-TMz-in-time-Ez");
        dumperEz.dumpGrid (Ez);

        BMPDumper<GridCoordinate3D> dumperHx;
        dumperHx.init (t, CURRENT, processId, "2D-TMz-in-time-Hx");
        dumperHx.dumpGrid (Hx);

        BMPDumper<GridCoordinate3D> dumperHy;
        dumperHy.init (t, CURRENT, processId, "2D-TMz-in-time-Hy");
        dumperHy.dumpGrid (Hy);
      }
    }
  }

  if (dumpRes)
  {
    /*
     * FIXME: leave only one dumper
     */
    BMPDumper<GridCoordinate3D> dumperEx;
    dumperEx.init (t, CURRENT, processId, "3D-amplitude-Ex");
    dumperEx.dumpGrid (ExAmplitude);

    BMPDumper<GridCoordinate3D> dumperEy;
    dumperEy.init (t, CURRENT, processId, "3D-amplitude-Ey");
    dumperEy.dumpGrid (EyAmplitude);

    BMPDumper<GridCoordinate3D> dumperEz;
    dumperEz.init (t, CURRENT, processId, "3D-amplitude-Ez");
    dumperEz.dumpGrid (EzAmplitude);

    BMPDumper<GridCoordinate3D> dumperHx;
    dumperHx.init (t, CURRENT, processId, "3D-amplitude-Hx");
    dumperHx.dumpGrid (HxAmplitude);

    BMPDumper<GridCoordinate3D> dumperHy;
    dumperHy.init (t, CURRENT, processId, "3D-amplitude-Hy");
    dumperHy.dumpGrid (HyAmplitude);

    BMPDumper<GridCoordinate3D> dumperHz;
    dumperHz.init (t, CURRENT, processId, "3D-amplitude-Hz");
    dumperHz.dumpGrid (HzAmplitude);
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.\n");
  }

#endif /* !COMPLEX_FIELD_VALUES */
}

int
Scheme3D::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
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
Scheme3D::performSteps (int dumpRes)
{
#if defined (CUDA_ENABLED)

#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
  int processId = 0;
#endif /* !PARALLEL_GRID */

  if (usePML || useTFSF || calculateAmplitude || useMetamaterials)
  {
    ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");
  }

  CudaExitStatus status;

  cudaExecute3DSteps (&status, yeeLayout, gridTimeStep, gridStep, Ex, Ey, Ez, Hx, Hy, Hz, Eps, Mu, totalStep, processId);

  ASSERT (status == CUDA_OK);

  if (dumpRes)
  {
    BMPDumper<GridCoordinate3D> dumper;
    dumper.init (totalStep, ALL, processId, "3D-TMz-in-time");
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
Scheme3D::initScheme (FPValue dx, FPValue sourceFreq)
{
  sourceFrequency = sourceFreq;
  sourceWaveLength = PhysicsConst::SpeedOfLight / sourceFrequency;

  gridStep = dx;
  courantNum = 2.0;
  gridTimeStep = gridStep / (courantNum * PhysicsConst::SpeedOfLight);
}

void
Scheme3D::initGrids ()
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
      for (int k = 0; k < Eps.getSize ().getZ (); ++k)
      {
        FieldPointValue* eps = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        eps->setCurValue (FieldValue (1, 0));
#else /* COMPLEX_FIELD_VALUES */
        eps->setCurValue (1);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);

        Eps.setFieldPointValue (eps, pos);
      }
    }
  }

  BMPDumper<GridCoordinate3D> dumper;
  dumper.init (0, CURRENT, processId, "Eps");
  dumper.dumpGrid (Eps);

  for (int i = 0; i < OmegaPE.getSize ().getX (); ++i)
  {
    for (int j = 0; j < OmegaPE.getSize ().getY (); ++j)
    {
      for (int k = 0; k < OmegaPE.getSize ().getZ (); ++k)
      {
        FieldPointValue* valOmega = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        valOmega->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
        valOmega->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);
        GridCoordinateFP3D posAbs = yeeLayout.getEpsCoordFP (OmegaPE.getTotalPosition (pos));

        GridCoordinateFP3D size = yeeLayout.getEpsCoordFP (OmegaPE.getTotalSize ());

        if (posAbs.getX () >= size.getX () / 4 && posAbs.getX () < 3*size.getX ()/4
            && posAbs.getY () >= yeeLayout.getLeftBorderPML ().getY () && posAbs.getY () < size.getY () - yeeLayout.getLeftBorderPML ().getY ()
            && posAbs.getZ () >= yeeLayout.getLeftBorderPML ().getZ () && posAbs.getZ () < size.getZ () - yeeLayout.getLeftBorderPML ().getZ ())
        {

//         if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
//             + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2)
//             + (posAbs.getZ () - size.getZ () / 2) * (posAbs.getZ () - size.getZ () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
//         {
#ifdef COMPLEX_FIELD_VALUES
          valOmega->setCurValue (FieldValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency, 0));
#else /* COMPLEX_FIELD_VALUES */
          valOmega->setCurValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency);
#endif /* !COMPLEX_FIELD_VALUES */
        }

        OmegaPE.setFieldPointValue (valOmega, pos);
      }
    }
  }

  for (int i = 0; i < OmegaPM.getSize ().getX (); ++i)
  {
    for (int j = 0; j < OmegaPM.getSize ().getY (); ++j)
    {
      for (int k = 0; k < OmegaPM.getSize ().getZ (); ++k)
      {
        FieldPointValue* valOmega = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        valOmega->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
        valOmega->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);
        GridCoordinateFP3D posAbs = yeeLayout.getEpsCoordFP (OmegaPM.getTotalPosition (pos));

        GridCoordinateFP3D size = yeeLayout.getEpsCoordFP (OmegaPM.getTotalSize ());

        if (posAbs.getX () >= size.getX () / 4 && posAbs.getX () < 3*size.getX ()/4
            && posAbs.getY () >= yeeLayout.getLeftBorderPML ().getY () && posAbs.getY () < size.getY () - yeeLayout.getLeftBorderPML ().getY ()
            && posAbs.getZ () >= yeeLayout.getLeftBorderPML ().getZ () && posAbs.getZ () < size.getZ () - yeeLayout.getLeftBorderPML ().getZ ())
        {

//         if ((posAbs.getX () - size.getX () / 2) * (posAbs.getX () - size.getX () / 2)
//             + (posAbs.getY () - size.getY () / 2) * (posAbs.getY () - size.getY () / 2)
//             + (posAbs.getZ () - size.getZ () / 2) * (posAbs.getZ () - size.getZ () / 2) < (size.getX ()*1.5/7.0) * (size.getX ()*1.5/7.0))
//         {
#ifdef COMPLEX_FIELD_VALUES
          valOmega->setCurValue (FieldValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency, 0));
#else /* COMPLEX_FIELD_VALUES */
          valOmega->setCurValue (sqrtf(2.0) * 2 * PhysicsConst::Pi * sourceFrequency);
#endif /* !COMPLEX_FIELD_VALUES */
        }

        OmegaPM.setFieldPointValue (valOmega, pos);
      }
    }
  }

  for (int i = 0; i < GammaE.getSize ().getX (); ++i)
  {
    for (int j = 0; j < GammaE.getSize ().getY (); ++j)
    {
      for (int k = 0; k < GammaE.getSize ().getZ (); ++k)
      {
        FieldPointValue* valGamma = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        valGamma->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
        valGamma->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);
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
  }

  for (int i = 0; i < GammaM.getSize ().getX (); ++i)
  {
    for (int j = 0; j < GammaM.getSize ().getY (); ++j)
    {
      for (int k = 0; k < GammaM.getSize ().getZ (); ++k)
      {
        FieldPointValue* valGamma = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        valGamma->setCurValue (FieldValue (0, 0));
#else /* COMPLEX_FIELD_VALUES */
        valGamma->setCurValue (0);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);
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

        GammaM.setFieldPointValue (valGamma, pos);
      }
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
      for (int k = 0; k < Mu.getSize ().getZ (); ++k)
      {
        FieldPointValue* mu = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
        mu->setCurValue (FieldValue (1, 0));
#else /* COMPLEX_FIELD_VALUES */
        mu->setCurValue (1);
#endif /* !COMPLEX_FIELD_VALUES */

        GridCoordinate3D pos (i, j, k);

        Mu.setFieldPointValue (mu, pos);
      }
    }
  }

  dumper.init (0, CURRENT, processId, "Mu");
  dumper.dumpGrid (Mu);

  FPValue eps0 = PhysicsConst::Eps0;
  FPValue mu0 = PhysicsConst::Mu0;

  GridCoordinate3D PMLSize = yeeLayout.getLeftBorderPML ();

  FPValue boundary = PMLSize.getX () * gridStep;
  uint32_t exponent = 6;
	FPValue R_err = 1e-16;
	FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
	FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

  for (int i = 0; i < SigmaX.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaX.getSize ().getY (); ++j)
    {
      for (int k = 0; k < SigmaX.getSize ().getZ (); ++k)
      {
        FieldPointValue* valSigma = new FieldPointValue ();

        GridCoordinate3D pos (i, j, k);
        GridCoordinateFP3D posAbs = yeeLayout.getEpsCoordFP (SigmaX.getTotalPosition (pos));

        GridCoordinateFP3D size = yeeLayout.getEpsCoordFP (SigmaX.getTotalSize ());

        /*
         * FIXME: add layout coordinates for material: sigma, eps, etc.
         */
        if (posAbs.getX () < PMLSize.getX ())
        {
          grid_coord dist = PMLSize.getX () - posAbs.getX ();
    			FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
    			FPValue x2 = dist * gridStep;       // lower bounds for point i

          FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));    //   polynomial grading

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
  }

  for (int i = 0; i < SigmaY.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaY.getSize ().getY (); ++j)
    {
      for (int k = 0; k < SigmaY.getSize ().getZ (); ++k)
      {
        FieldPointValue* valSigma = new FieldPointValue ();

        GridCoordinate3D pos (i, j, k);
        GridCoordinateFP3D posAbs = yeeLayout.getEpsCoordFP (SigmaY.getTotalPosition (pos));

        GridCoordinateFP3D size = yeeLayout.getEpsCoordFP (SigmaY.getTotalSize ());

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
  }

  for (int i = 0; i < SigmaZ.getSize ().getX (); ++i)
  {
    for (int j = 0; j < SigmaZ.getSize ().getY (); ++j)
    {
      for (int k = 0; k < SigmaZ.getSize ().getZ (); ++k)
      {
        FieldPointValue* valSigma = new FieldPointValue ();

        GridCoordinate3D pos (i, j, k);
        GridCoordinateFP3D posAbs = yeeLayout.getEpsCoordFP (SigmaZ.getTotalPosition (pos));

        GridCoordinateFP3D size = yeeLayout.getEpsCoordFP (SigmaZ.getTotalSize ());

        /*
         * FIXME: add layout coordinates for material: sigma, eps, etc.
         */
        if (posAbs.getZ () < PMLSize.getZ ())
        {
          grid_coord dist = PMLSize.getZ () - posAbs.getZ ();
          FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
          FPValue x2 = dist * gridStep;       // lower bounds for point i

          FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

#ifdef COMPLEX_FIELD_VALUES
    			valSigma->setCurValue (FieldValue (val, 0));
#else /* COMPLEX_FIELD_VALUES */
          valSigma->setCurValue (val);
#endif /* !COMPLEX_FIELD_VALUES */
        }
        else if (posAbs.getZ () >= size.getZ () - PMLSize.getZ ())
        {
          grid_coord dist = posAbs.getZ () - (size.getZ () - PMLSize.getZ ());
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

        SigmaZ.setFieldPointValue (valSigma, pos);
      }
    }
  }

  dumper.init (0, CURRENT, processId, "SigmaX");
  dumper.dumpGrid (SigmaX);
  dumper.init (0, CURRENT, processId, "SigmaY");
  dumper.dumpGrid (SigmaY);
  dumper.init (0, CURRENT, processId, "SigmaZ");
  dumper.dumpGrid (SigmaZ);

  for (int i = 0; i < Ex.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ex.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ex.getSize ().getZ (); ++k)
      {
        FieldPointValue* valEx = new FieldPointValue ();

        FieldPointValue* valDx = new FieldPointValue ();

        FieldPointValue* valD1x = new FieldPointValue ();

        FieldPointValue* valExAmp;
        if (calculateAmplitude)
        {
          valExAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Ex.setFieldPointValue (valEx, pos);

        Dx.setFieldPointValue (valDx, pos);

        D1x.setFieldPointValue (valD1x, pos);

        if (calculateAmplitude)
        {
          ExAmplitude.setFieldPointValue (valExAmp, pos);
        }
      }
    }
  }

  for (int i = 0; i < Ey.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ey.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ey.getSize ().getZ (); ++k)
      {
        FieldPointValue* valEy = new FieldPointValue ();

        FieldPointValue* valDy = new FieldPointValue ();

        FieldPointValue* valD1y = new FieldPointValue ();

        FieldPointValue* valEyAmp;
        if (calculateAmplitude)
        {
          valEyAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Ey.setFieldPointValue (valEy, pos);

        Dy.setFieldPointValue (valDy, pos);

        D1y.setFieldPointValue (valD1y, pos);

        if (calculateAmplitude)
        {
          EyAmplitude.setFieldPointValue (valEyAmp, pos);
        }
      }
    }
  }

  for (int i = 0; i < Ez.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ez.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ez.getSize ().getZ (); ++k)
      {
        FieldPointValue* valEz = new FieldPointValue ();

        FieldPointValue* valDz = new FieldPointValue ();

        FieldPointValue* valD1z = new FieldPointValue ();

        FieldPointValue* valEzAmp;
        if (calculateAmplitude)
        {
          valEzAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Ez.setFieldPointValue (valEz, pos);

        Dz.setFieldPointValue (valDz, pos);

        D1z.setFieldPointValue (valD1z, pos);

        if (calculateAmplitude)
        {
          EzAmplitude.setFieldPointValue (valEzAmp, pos);
        }
      }
    }
  }

  for (int i = 0; i < Hx.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hx.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hx.getSize ().getZ (); ++k)
      {
        FieldPointValue* valHx = new FieldPointValue ();

        FieldPointValue* valBx = new FieldPointValue ();

        FieldPointValue* valB1x = new FieldPointValue ();

        FieldPointValue* valHxAmp;
        if (calculateAmplitude)
        {
          valHxAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Hx.setFieldPointValue (valHx, pos);

        Bx.setFieldPointValue (valBx, pos);

        B1x.setFieldPointValue (valB1x, pos);

        if (calculateAmplitude)
        {
          HxAmplitude.setFieldPointValue (valHxAmp, pos);
        }
      }
    }
  }

  for (int i = 0; i < Hy.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hy.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hy.getSize ().getZ (); ++k)
      {
        FieldPointValue* valHy = new FieldPointValue ();

        FieldPointValue* valBy = new FieldPointValue ();

        FieldPointValue* valB1y = new FieldPointValue ();

        FieldPointValue* valHyAmp;
        if (calculateAmplitude)
        {
          valHyAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Hy.setFieldPointValue (valHy, pos);

        By.setFieldPointValue (valBy, pos);

        B1y.setFieldPointValue (valB1y, pos);

        if (calculateAmplitude)
        {
          HyAmplitude.setFieldPointValue (valHyAmp, pos);
        }
      }
    }
  }

  for (int i = 0; i < Hz.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hz.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hz.getSize ().getZ (); ++k)
      {
        FieldPointValue* valHz = new FieldPointValue ();

        FieldPointValue* valBz = new FieldPointValue ();

        FieldPointValue* valB1z = new FieldPointValue ();

        FieldPointValue* valHzAmp;
        if (calculateAmplitude)
        {
          valHzAmp = new FieldPointValue ();
        }

        GridCoordinate3D pos (i, j, k);

        Hz.setFieldPointValue (valHz, pos);

        Bz.setFieldPointValue (valBz, pos);

        B1z.setFieldPointValue (valB1z, pos);

        if (calculateAmplitude)
        {
          HzAmplitude.setFieldPointValue (valHzAmp, pos);
        }
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

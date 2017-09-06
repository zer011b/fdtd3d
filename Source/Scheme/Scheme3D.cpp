#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "Kernels.h"
#include "Settings.h"
#include "Scheme3D.h"
#include "Approximation.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_3D

#define DO_USE_3D_MODE (true)

Scheme3D::Scheme3D (YeeGridLayout *layout,
                    const GridCoordinate3D& totSize,
                    time_step tStep)
  : yeeLayout (layout)
  , Ex (NULLPTR)
  , Ey (NULLPTR)
  , Ez (NULLPTR)
  , Hx (NULLPTR)
  , Hy (NULLPTR)
  , Hz (NULLPTR)
  , Dx (NULLPTR)
  , Dy (NULLPTR)
  , Dz (NULLPTR)
  , Bx (NULLPTR)
  , By (NULLPTR)
  , Bz (NULLPTR)
  , D1x (NULLPTR)
  , D1y (NULLPTR)
  , D1z (NULLPTR)
  , B1x (NULLPTR)
  , B1y (NULLPTR)
  , B1z (NULLPTR)
  , ExAmplitude (NULLPTR)
  , EyAmplitude (NULLPTR)
  , EzAmplitude (NULLPTR)
  , HxAmplitude (NULLPTR)
  , HyAmplitude (NULLPTR)
  , HzAmplitude (NULLPTR)
  , Eps (NULLPTR)
  , Mu (NULLPTR)
  , OmegaPE (NULLPTR)
  , OmegaPM (NULLPTR)
  , GammaE (NULLPTR)
  , GammaM (NULLPTR)
  , SigmaX (NULLPTR)
  , SigmaY (NULLPTR)
  , SigmaZ (NULLPTR)
  , EInc (NULLPTR)
  , HInc (NULLPTR)
  , totalEx (NULLPTR)
  , totalEy (NULLPTR)
  , totalEz (NULLPTR)
  , totalHx (NULLPTR)
  , totalHy (NULLPTR)
  , totalHz (NULLPTR)
  , totalInitialized (false)
  , totalEps (NULLPTR)
  , totalMu (NULLPTR)
  , totalOmegaPE (NULLPTR)
  , totalOmegaPM (NULLPTR)
  , totalGammaE (NULLPTR)
  , totalGammaM (NULLPTR)
  , sourceWaveLength (0)
  , sourceFrequency (0)
  , courantNum (0)
  , gridStep (0)
  , gridTimeStep (0)
  , totalStep (tStep)
  , leftNTFF (GridCoordinate3D (solverSettings.getNTFFSizeX (), solverSettings.getNTFFSizeY (), solverSettings.getNTFFSizeZ ()))
  , rightNTFF (layout->getEzSize () - leftNTFF + GridCoordinate3D (1,1,1))
{
  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    GridCoordinate3D bufSize (solverSettings.getBufferSize (),
                              solverSettings.getBufferSize (),
                              solverSettings.getBufferSize ());

    ParallelYeeGridLayout *parallelYeeLayout = (ParallelYeeGridLayout *) (layout);

    Eps = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "Eps");
    Mu = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getMuSizeForCurNode (), "Mu");

    Ex = new ParallelGrid (parallelYeeLayout->getExSize (), bufSize, 0, parallelYeeLayout->getExSizeForCurNode (), "Ex");
    Ey = new ParallelGrid (parallelYeeLayout->getEySize (), bufSize, 0, parallelYeeLayout->getEySizeForCurNode (), "Ey");
    Ez = new ParallelGrid (parallelYeeLayout->getEzSize (), bufSize, 0, parallelYeeLayout->getEzSizeForCurNode (), "Ez");
    Hx = new ParallelGrid (parallelYeeLayout->getHxSize (), bufSize, 0, parallelYeeLayout->getHxSizeForCurNode (), "Hx");
    Hy = new ParallelGrid (parallelYeeLayout->getHySize (), bufSize, 0, parallelYeeLayout->getHySizeForCurNode (), "Hy");
    Hz = new ParallelGrid (parallelYeeLayout->getHzSize (), bufSize, 0, parallelYeeLayout->getHzSizeForCurNode (), "Hz");

    if (solverSettings.getDoUsePML ())
    {
      Dx = new ParallelGrid (parallelYeeLayout->getExSize (), bufSize, 0, parallelYeeLayout->getExSizeForCurNode (), "Dx");
      Dy = new ParallelGrid (parallelYeeLayout->getEySize (), bufSize, 0, parallelYeeLayout->getEySizeForCurNode (), "Dy");
      Dz = new ParallelGrid (parallelYeeLayout->getEzSize (), bufSize, 0, parallelYeeLayout->getEzSizeForCurNode (), "Dz");
      Bx = new ParallelGrid (parallelYeeLayout->getHxSize (), bufSize, 0, parallelYeeLayout->getHxSizeForCurNode (), "Bx");
      By = new ParallelGrid (parallelYeeLayout->getHySize (), bufSize, 0, parallelYeeLayout->getHySizeForCurNode (), "By");
      Bz = new ParallelGrid (parallelYeeLayout->getHzSize (), bufSize, 0, parallelYeeLayout->getHzSizeForCurNode (), "Bz");

      if (solverSettings.getDoUseMetamaterials ())
      {
        D1x = new ParallelGrid (parallelYeeLayout->getExSize (), bufSize, 0, parallelYeeLayout->getExSizeForCurNode (), "D1x");
        D1y = new ParallelGrid (parallelYeeLayout->getEySize (), bufSize, 0, parallelYeeLayout->getEySizeForCurNode (), "D1y");
        D1z = new ParallelGrid (parallelYeeLayout->getEzSize (), bufSize, 0, parallelYeeLayout->getEzSizeForCurNode (), "D1z");
        B1x = new ParallelGrid (parallelYeeLayout->getHxSize (), bufSize, 0, parallelYeeLayout->getHxSizeForCurNode (), "B1x");
        B1y = new ParallelGrid (parallelYeeLayout->getHySize (), bufSize, 0, parallelYeeLayout->getHySizeForCurNode (), "B1y");
        B1z = new ParallelGrid (parallelYeeLayout->getHzSize (), bufSize, 0, parallelYeeLayout->getHzSizeForCurNode (), "B1z");
      }

      SigmaX = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "SigmaX");
      SigmaY = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "SigmaY");
      SigmaZ = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "SigmaZ");
    }

    if (solverSettings.getDoUseAmplitudeMode ())
    {
      ExAmplitude = new ParallelGrid (parallelYeeLayout->getExSize (), bufSize, 0, parallelYeeLayout->getExSizeForCurNode (), "ExAmp");
      EyAmplitude = new ParallelGrid (parallelYeeLayout->getEySize (), bufSize, 0, parallelYeeLayout->getEySizeForCurNode (), "EyAmp");
      EzAmplitude = new ParallelGrid (parallelYeeLayout->getEzSize (), bufSize, 0, parallelYeeLayout->getEzSizeForCurNode (), "EzAmp");
      HxAmplitude = new ParallelGrid (parallelYeeLayout->getHxSize (), bufSize, 0, parallelYeeLayout->getHxSizeForCurNode (), "HxAmp");
      HyAmplitude = new ParallelGrid (parallelYeeLayout->getHySize (), bufSize, 0, parallelYeeLayout->getHySizeForCurNode (), "HyAmp");
      HzAmplitude = new ParallelGrid (parallelYeeLayout->getHzSize (), bufSize, 0, parallelYeeLayout->getHzSizeForCurNode (), "HzAmp");
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      OmegaPE = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "OmegaPE");
      GammaE = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "GammaE");
      OmegaPM = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "OmegaPM");
      GammaM = new ParallelGrid (parallelYeeLayout->getEpsSize (), bufSize, 0, parallelYeeLayout->getEpsSizeForCurNode (), "GammaM");
    }

    if (!solverSettings.getEpsFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalEps = new Grid<GridCoordinate3D> (parallelYeeLayout->getEpsSize (), 0, "totalEps");
    }
    if (!solverSettings.getMuFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalMu = new Grid<GridCoordinate3D> (parallelYeeLayout->getMuSize (), 0, "totalMu");
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (!solverSettings.getOmegaPEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPE = new Grid<GridCoordinate3D> (parallelYeeLayout->getEpsSize (), 0, "totalOmegaPE");
      }
      if (!solverSettings.getOmegaPMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPM = new Grid<GridCoordinate3D> (parallelYeeLayout->getEpsSize (), 0, "totalOmegaPM");
      }
      if (!solverSettings.getGammaEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaE = new Grid<GridCoordinate3D> (parallelYeeLayout->getEpsSize (), 0, "totalGammaE");
      }
      if (!solverSettings.getGammaMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaM = new Grid<GridCoordinate3D> (parallelYeeLayout->getEpsSize (), 0, "totalGammaM");
      }
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }
  else
  {
    Eps = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "Eps");
    Mu = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "Mu");

    Ex = new Grid<GridCoordinate3D> (layout->getExSize (), 0, "Ex");
    Ey = new Grid<GridCoordinate3D> (layout->getEySize (), 0, "Ey");
    Ez = new Grid<GridCoordinate3D> (layout->getEzSize (), 0, "Ez");
    Hx = new Grid<GridCoordinate3D> (layout->getHxSize (), 0, "Hx");
    Hy = new Grid<GridCoordinate3D> (layout->getHySize (), 0, "Hy");
    Hz = new Grid<GridCoordinate3D> (layout->getHzSize (), 0, "Hz");

    if (solverSettings.getDoUsePML ())
    {
      Dx = new Grid<GridCoordinate3D> (layout->getExSize (), 0, "Dx");
      Dy = new Grid<GridCoordinate3D> (layout->getEySize (), 0, "Dy");
      Dz = new Grid<GridCoordinate3D> (layout->getEzSize (), 0, "Dz");
      Bx = new Grid<GridCoordinate3D> (layout->getHxSize (), 0, "Bx");
      By = new Grid<GridCoordinate3D> (layout->getHySize (), 0, "By");
      Bz = new Grid<GridCoordinate3D> (layout->getHzSize (), 0, "Bz");

      if (solverSettings.getDoUseMetamaterials ())
      {
        D1x = new Grid<GridCoordinate3D> (layout->getExSize (), 0, "D1x");
        D1y = new Grid<GridCoordinate3D> (layout->getEySize (), 0, "D1y");
        D1z = new Grid<GridCoordinate3D> (layout->getEzSize (), 0, "D1z");
        B1x = new Grid<GridCoordinate3D> (layout->getHxSize (), 0, "B1x");
        B1y = new Grid<GridCoordinate3D> (layout->getHySize (), 0, "B1y");
        B1z = new Grid<GridCoordinate3D> (layout->getHzSize (), 0, "B1z");
      }

      SigmaX = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "SigmaX");
      SigmaY = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "SigmaY");
      SigmaZ = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "SigmaZ");
    }

    if (solverSettings.getDoUseAmplitudeMode ())
    {
      ExAmplitude = new Grid<GridCoordinate3D> (layout->getExSize (), 0, "ExAmp");
      EyAmplitude = new Grid<GridCoordinate3D> (layout->getEySize (), 0, "EyAmp");
      EzAmplitude = new Grid<GridCoordinate3D> (layout->getEzSize (), 0, "EzAmp");
      HxAmplitude = new Grid<GridCoordinate3D> (layout->getHxSize (), 0, "HxAmp");
      HyAmplitude = new Grid<GridCoordinate3D> (layout->getHySize (), 0, "HyAmp");
      HzAmplitude = new Grid<GridCoordinate3D> (layout->getHzSize (), 0, "HzAmp");
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      OmegaPE = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "OmegaPE");
      GammaE = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "GammaE");
      OmegaPM = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "OmegaPM");
      GammaM = new Grid<GridCoordinate3D> (layout->getEpsSize (), 0, "GammaM");
    }

    totalEps = Eps;
    totalMu = Mu;
    totalOmegaPE = OmegaPE;
    totalOmegaPM = OmegaPM;
    totalGammaE = GammaE;
    totalGammaM = GammaM;
  }

  if (solverSettings.getDoUseTFSF ())
  {
    EInc = new Grid<GridCoordinate1D> (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0, "EInc");
    HInc = new Grid<GridCoordinate1D> (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0, "HInc");
  }

  ASSERT (!solverSettings.getDoUseTFSF ()
          || (solverSettings.getDoUseTFSF () && yeeLayout->getSizeTFSF () != GridCoordinate3D (0, 0, 0)));

  ASSERT (!solverSettings.getDoUsePML ()
          || (solverSettings.getDoUsePML () && (yeeLayout->getSizePML () != GridCoordinate3D (0, 0, 0))));

  ASSERT (!solverSettings.getDoUseAmplitudeMode ()
          || solverSettings.getDoUseAmplitudeMode () && solverSettings.getNumAmplitudeSteps () != 0);

#ifdef COMPLEX_FIELD_VALUES
  ASSERT (!solverSettings.getDoUseAmplitudeMode ());
#endif /* COMPLEX_FIELD_VALUES */

  if (solverSettings.getDoSaveAsBMP ())
  {
    dumper[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate3D> ();

    PaletteType palette = PaletteType::PALETTE_GRAY;
    OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

    if (solverSettings.getDoUsePaletteGray ())
    {
      palette = PaletteType::PALETTE_GRAY;
    }
    else if (solverSettings.getDoUsePaletteRGB ())
    {
      palette = PaletteType::PALETTE_BLUE_GREEN_RED;
    }
    else
    {
      UNREACHABLE;
    }

    if (solverSettings.getDoUseOrthAxisX ())
    {
      orthogonalAxis = OrthogonalAxis::X;
    }
    else if (solverSettings.getDoUseOrthAxisY ())
    {
      orthogonalAxis = OrthogonalAxis::Y;
    }
    else if (solverSettings.getDoUseOrthAxisZ ())
    {
      orthogonalAxis = OrthogonalAxis::Z;
    }
    else
    {
      UNREACHABLE;
    }

    ((BMPDumper<GridCoordinate3D> *) dumper[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  else
  {
    dumper[FILE_TYPE_BMP] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsDAT ())
  {
    dumper[FILE_TYPE_DAT] = new DATDumper<GridCoordinate3D> ();
  }
  else
  {
    dumper[FILE_TYPE_DAT] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsTXT ())
  {
    dumper[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate3D> ();
  }
  else
  {
    dumper[FILE_TYPE_TXT] = NULLPTR;
  }

  {
    loader[FILE_TYPE_BMP] = new BMPLoader<GridCoordinate3D> ();

    PaletteType palette = PaletteType::PALETTE_GRAY;
    OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

    if (solverSettings.getDoUsePaletteGray ())
    {
      palette = PaletteType::PALETTE_GRAY;
    }
    else if (solverSettings.getDoUsePaletteRGB ())
    {
      palette = PaletteType::PALETTE_BLUE_GREEN_RED;
    }
    else
    {
      UNREACHABLE;
    }

    if (solverSettings.getDoUseOrthAxisX ())
    {
      orthogonalAxis = OrthogonalAxis::X;
    }
    else if (solverSettings.getDoUseOrthAxisY ())
    {
      orthogonalAxis = OrthogonalAxis::Y;
    }
    else if (solverSettings.getDoUseOrthAxisZ ())
    {
      orthogonalAxis = OrthogonalAxis::Z;
    }
    else
    {
      UNREACHABLE;
    }

    ((BMPLoader<GridCoordinate3D> *) loader[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  {
    loader[FILE_TYPE_DAT] = new DATLoader<GridCoordinate3D> ();
  }
  {
    loader[FILE_TYPE_TXT] = new TXTLoader<GridCoordinate3D> ();
  }
}

Scheme3D::~Scheme3D ()
{
  delete Eps;
  delete Mu;

  delete Ex;
  delete Ey;
  delete Ez;

  delete Hx;
  delete Hy;
  delete Hz;

  if (solverSettings.getDoUsePML ())
  {
    delete Dx;
    delete Dy;
    delete Dz;

    delete Bx;
    delete By;
    delete Bz;

    if (solverSettings.getDoUseMetamaterials ())
    {
      delete D1x;
      delete D1y;
      delete D1z;

      delete B1x;
      delete B1y;
      delete B1z;
    }

    delete SigmaX;
    delete SigmaY;
    delete SigmaZ;
  }

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    delete ExAmplitude;
    delete EyAmplitude;
    delete EzAmplitude;
    delete HxAmplitude;
    delete HyAmplitude;
    delete HzAmplitude;
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
    delete OmegaPE;
    delete OmegaPM;
    delete GammaE;
    delete GammaM;
  }

  if (solverSettings.getDoUseTFSF ())
  {
    delete EInc;
    delete HInc;
  }

  if (totalInitialized)
  {
    delete totalEx;
    delete totalEy;
    delete totalEz;

    delete totalHx;
    delete totalHy;
    delete totalHz;
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    delete totalEps;
    delete totalMu;

    delete totalOmegaPE;
    delete totalOmegaPM;
    delete totalGammaE;
    delete totalGammaM;
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  delete dumper[FILE_TYPE_BMP];
  delete dumper[FILE_TYPE_DAT];
  delete dumper[FILE_TYPE_TXT];

  delete loader[FILE_TYPE_BMP];
  delete loader[FILE_TYPE_DAT];
  delete loader[FILE_TYPE_TXT];
}

void
Scheme3D::performPlaneWaveESteps (time_step t)
{
  grid_coord size = EInc->getSize ().getX ();

  ASSERT (size > 0);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Eps0 * gridStep);

  for (grid_coord i = 1; i < size; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valE = EInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i - 1);
    GridCoordinate1D posRight (i);

    FieldPointValue *valH1 = HInc->getFieldPointValue (posLeft);
    FieldPointValue *valH2 = HInc->getFieldPointValue (posRight);

    FieldValue val = valE->getPrevValue () + modifier * (valH1->getPrevValue () - valH2->getPrevValue ());

    valE->setCurValue (val);
  }

  GridCoordinate1D pos (0);
  FieldPointValue *valE = EInc->getFieldPointValue (pos);

#ifdef COMPLEX_FIELD_VALUES
  valE->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                 cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
  valE->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */

  ASSERT (EInc->getFieldPointValue (GridCoordinate1D (size - 1))->getCurValue () == getFieldValueRealOnly (0.0));

  EInc->nextTimeStep ();
}

void
Scheme3D::performPlaneWaveHSteps (time_step t)
{
  grid_coord size = HInc->getSize ().getX ();

  ASSERT (size > 1);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Mu0 * gridStep);

  for (grid_coord i = 0; i < size - 1; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valH = HInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i);
    GridCoordinate1D posRight (i + 1);

    FieldPointValue *valE1 = EInc->getFieldPointValue (posLeft);
    FieldPointValue *valE2 = EInc->getFieldPointValue (posRight);

    FieldValue val = valH->getPrevValue () + modifier * (valE1->getPrevValue () - valE2->getPrevValue ());

    valH->setCurValue (val);
  }

  ASSERT (HInc->getFieldPointValue (GridCoordinate1D (size - 2))->getCurValue () == getFieldValueRealOnly (0.0));

  HInc->nextTimeStep ();
}

void
Scheme3D::performExSteps (time_step t, GridCoordinate3D ExStart, GridCoordinate3D ExEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (solverSettings.getDoUsePML ())
  {
    calculateExStepPML (t, ExStart, ExEnd);
  }
  else
  {
    calculateExStep (t, ExStart, ExEnd);
  }

  if (solverSettings.getDoUsePointSourceEx ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::EX)> (t);
  }
}

FieldValue
Scheme3D::approximateIncidentWave (GridCoordinateFP3D realCoord, FPValue dDiff, Grid<GridCoordinate1D> &FieldInc)
{
  GridCoordinateFP3D zeroCoordFP = yeeLayout->getZeroIncCoordFP ();

  FPValue x = realCoord.getX () - zeroCoordFP.getX ();
  FPValue y = realCoord.getY () - zeroCoordFP.getY ();
  FPValue z = realCoord.getZ () - zeroCoordFP.getZ ();
  FPValue d = x * sin (yeeLayout->getIncidentWaveAngle1 ()) * cos (yeeLayout->getIncidentWaveAngle2 ())
              + y * sin (yeeLayout->getIncidentWaveAngle1 ()) * sin (yeeLayout->getIncidentWaveAngle2 ())
              + z * cos (yeeLayout->getIncidentWaveAngle1 ()) - dDiff;
  FPValue coordD1 = (FPValue) ((grid_iter) d);
  FPValue coordD2 = coordD1 + 1;
  FPValue proportionD2 = d - coordD1;
  FPValue proportionD1 = 1 - proportionD2;

  GridCoordinate1D pos1 (coordD1);
  GridCoordinate1D pos2 (coordD2);

  FieldPointValue *val1 = FieldInc.getFieldPointValue (pos1);
  FieldPointValue *val2 = FieldInc.getFieldPointValue (pos2);

  return proportionD1 * val1->getPrevValue () + proportionD2 * val2->getPrevValue ();
}

FieldValue
Scheme3D::approximateIncidentWaveE (GridCoordinateFP3D realCoord)
{
  return approximateIncidentWave (realCoord, 0.0, *EInc);
}

FieldValue
Scheme3D::approximateIncidentWaveH (GridCoordinateFP3D realCoord)
{
  return approximateIncidentWave (realCoord, 0.5, *HInc);
}

void
Scheme3D::calculateExTFSF (GridCoordinate3D posAbs,
                           FieldValue &valHz1,
                           FieldValue &valHz2,
                           FieldValue &valHy1,
                           FieldValue &valHy2,
                           GridCoordinate3D posDown,
                           GridCoordinate3D posUp,
                           GridCoordinate3D posBack,
                           GridCoordinate3D posFront)
{
  bool do_need_update_down = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, DO_USE_3D_MODE);
  bool do_need_update_up = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, DO_USE_3D_MODE);

  bool do_need_update_back = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK, DO_USE_3D_MODE);
  bool do_need_update_front = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT, DO_USE_3D_MODE);

  GridCoordinate3D auxPosY;
  GridCoordinate3D auxPosZ;
  FieldValue diffY;
  FieldValue diffZ;

  if (do_need_update_down)
  {
    auxPosY = posUp;
  }
  else if (do_need_update_up)
  {
    auxPosY = posDown;
  }

  if (do_need_update_back)
  {
    auxPosZ = posFront;
  }
  else if (do_need_update_front)
  {
    auxPosZ = posBack;
  }

  if (do_need_update_down || do_need_update_up)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPosY));

    diffY = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_back || do_need_update_front)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPosZ));

    diffZ = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_down)
  {
    valHz1 -= diffY;
  }
  else if (do_need_update_up)
  {
    valHz2 -= diffY;
  }

  if (do_need_update_back)
  {
    valHy1 -= diffZ;
  }
  else if (do_need_update_front)
  {
    valHy2 -= diffZ;
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
        GridCoordinate3D posAbs = Ex->getTotalPosition (pos);

        FieldPointValue* valEx = Ex->getFieldPointValue (pos);

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::EX, Eps, GridType::EPS);

        GridCoordinate3D posDown = yeeLayout->getExCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getExCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout->getExCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getExCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHz1 = Hz->getFieldPointValue (posUp);
        FieldPointValue* valHz2 = Hz->getFieldPointValue (posDown);

        FieldPointValue* valHy1 = Hy->getFieldPointValue (posFront);
        FieldPointValue* valHy2 = Hy->getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateExTFSF (posAbs, prevHz1, prevHz2, prevHy1, prevHy2, posDown, posUp, posBack, posFront);
        }

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
        GridCoordinate3D posAbs = Dx->getTotalPosition (pos);

        FieldPointValue* valDx = Dx->getFieldPointValue (pos);

        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::DX, SigmaY, GridType::SIGMAY);

        GridCoordinate3D posDown = yeeLayout->getExCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getExCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout->getExCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getExCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHz1 = Hz->getFieldPointValue (posUp);
        FieldPointValue* valHz2 = Hz->getFieldPointValue (posDown);

        FieldPointValue* valHy1 = Hy->getFieldPointValue (posFront);
        FieldPointValue* valHy2 = Hy->getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateExTFSF (posAbs, prevHz1, prevHz2, prevHy1, prevHy2, posDown, posUp, posBack, posFront);
        }

        /*
         * FIXME: precalculate coefficients
         */
        FPValue k_y = 1;

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

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Dx->getTotalPosition (pos);

          FieldPointValue* valD1x = D1x->getFieldPointValue (pos);
          FieldPointValue* valDx = Dx->getFieldPointValue (pos);

          FPValue omegaPE;
          FPValue gammaE;
          FPValue eps = yeeLayout->getMetaMaterial (posAbs, GridType::DX, Eps, GridType::EPS, OmegaPE, GridType::OMEGAPE, GammaE, GridType::GAMMAE, omegaPE, gammaE);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
  {
    for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
    {
      for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Ex->getTotalPosition (pos);

        FieldPointValue* valEx = Ex->getFieldPointValue (pos);

        FieldPointValue* valDx;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valDx = D1x->getFieldPointValue (pos);
        }
        else
        {
          valDx = Dx->getFieldPointValue (pos);
        }

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::DX, Eps, GridType::EPS);
        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::DX, SigmaX, GridType::SIGMAX);
        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::DX, SigmaZ, GridType::SIGMAZ);

        FPValue modifier = eps * eps0;
        if (solverSettings.getDoUseMetamaterials ())
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
  if (solverSettings.getDoUsePML ())
  {
    calculateEyStepPML (t, EyStart, EyEnd);
  }
  else
  {
    calculateEyStep (t, EyStart, EyEnd);
  }

  if (solverSettings.getDoUsePointSourceEy ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::EY)> (t);
  }
}

void
Scheme3D::calculateEyTFSF (GridCoordinate3D posAbs,
                           FieldValue &valHz1,
                           FieldValue &valHz2,
                           FieldValue &valHx1,
                           FieldValue &valHx2,
                           GridCoordinate3D posLeft,
                           GridCoordinate3D posRight,
                           GridCoordinate3D posBack,
                           GridCoordinate3D posFront)
{
  bool do_need_update_left = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT, DO_USE_3D_MODE);
  bool do_need_update_right = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT, DO_USE_3D_MODE);

  bool do_need_update_back = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::BACK, DO_USE_3D_MODE);
  bool do_need_update_front = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::FRONT, DO_USE_3D_MODE);

  GridCoordinate3D auxPosX;
  GridCoordinate3D auxPosZ;
  FieldValue diffX;
  FieldValue diffZ;

  if (do_need_update_left)
  {
    auxPosX = posRight;
  }
  else if (do_need_update_right)
  {
    auxPosX = posLeft;
  }

  if (do_need_update_back)
  {
    auxPosZ = posFront;
  }
  else if (do_need_update_front)
  {
    auxPosZ = posBack;
  }

  if (do_need_update_left || do_need_update_right)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPosX));

    diffX = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_back || do_need_update_front)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPosZ));

    diffZ = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_left)
  {
    valHz1 -= diffX;
  }
  else if (do_need_update_right)
  {
    valHz2 -= diffX;
  }

  if (do_need_update_back)
  {
    valHx1 -= diffZ;
  }
  else if (do_need_update_front)
  {
    valHx2 -= diffZ;
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
        GridCoordinate3D posAbs = Ey->getTotalPosition (pos);

        FieldPointValue* valEy = Ey->getFieldPointValue (pos);

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::EY, Eps, GridType::EPS);

        GridCoordinate3D posLeft = yeeLayout->getEyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getEyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout->getEyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getEyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHz1 = Hz->getFieldPointValue (posRight);
        FieldPointValue* valHz2 = Hz->getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx->getFieldPointValue (posFront);
        FieldPointValue* valHx2 = Hx->getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateEyTFSF (posAbs, prevHz1, prevHz2, prevHx1, prevHx2, posLeft, posRight, posBack, posFront);
        }

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
        GridCoordinate3D posAbs = Dy->getTotalPosition (pos);

        FieldPointValue* valDy = Dy->getFieldPointValue (pos);

        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::DY, SigmaZ, GridType::SIGMAZ);

        GridCoordinate3D posLeft = yeeLayout->getEyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getEyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout->getEyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getEyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valHz1 = Hz->getFieldPointValue (posRight);
        FieldPointValue* valHz2 = Hz->getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx->getFieldPointValue (posFront);
        FieldPointValue* valHx2 = Hx->getFieldPointValue (posBack);

        FieldValue prevHz1 = valHz1->getPrevValue ();
        FieldValue prevHz2 = valHz2->getPrevValue ();

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateEyTFSF (posAbs, prevHz1, prevHz2, prevHx1, prevHx2, posLeft, posRight, posBack, posFront);
        }

        /*
         * FIXME: precalculate coefficients
         */
        FPValue k_z = 1;

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

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
    {
      for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
      {
        for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Dy->getTotalPosition (pos);

          FieldPointValue* valD1y = D1y->getFieldPointValue (pos);
          FieldPointValue* valDy = Dy->getFieldPointValue (pos);

          FPValue omegaPE;
          FPValue gammaE;
          FPValue eps = yeeLayout->getMetaMaterial (posAbs, GridType::DY, Eps, GridType::EPS, OmegaPE, GridType::OMEGAPE, GammaE, GridType::GAMMAE, omegaPE, gammaE);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
  {
    for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
    {
      for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Ey->getTotalPosition (pos);

        FieldPointValue* valEy = Ey->getFieldPointValue (pos);

        FieldPointValue* valDy;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valDy = D1y->getFieldPointValue (pos);
        }
        else
        {
          valDy = Dy->getFieldPointValue (pos);
        }

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::DY, Eps, GridType::EPS);
        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::DY, SigmaX, GridType::SIGMAX);
        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::DY, SigmaY, GridType::SIGMAY);

        FPValue modifier = eps * eps0;
        if (solverSettings.getDoUseMetamaterials ())
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
  if (solverSettings.getDoUsePML ())
  {
    calculateEzStepPML (t, EzStart, EzEnd);
  }
  else
  {
    calculateEzStep (t, EzStart, EzEnd);
  }

  if (solverSettings.getDoUsePointSourceEz ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::EZ)> (t);
  }
}

void
Scheme3D::calculateEzTFSF (GridCoordinate3D posAbs,
                           FieldValue &valHy1,
                           FieldValue &valHy2,
                           FieldValue &valHx1,
                           FieldValue &valHx2,
                           GridCoordinate3D posLeft,
                           GridCoordinate3D posRight,
                           GridCoordinate3D posDown,
                           GridCoordinate3D posUp)
{
  bool do_need_update_left = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT, DO_USE_3D_MODE);
  bool do_need_update_right = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT, DO_USE_3D_MODE);

  bool do_need_update_down = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN, DO_USE_3D_MODE);
  bool do_need_update_up = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP, DO_USE_3D_MODE);

  GridCoordinate3D auxPosX;
  GridCoordinate3D auxPosY;
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
    GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPosX));

    diffX = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_down || do_need_update_up)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPosY));

    diffY = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
  }

  if (do_need_update_left)
  {
    valHy1 -= diffX;
  }
  else if (do_need_update_right)
  {
    valHy2 -= diffX;
  }

  if (do_need_update_down)
  {
    valHx1 -= diffY;
  }
  else if (do_need_update_up)
  {
    valHx2 -= diffY;
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
        GridCoordinate3D posAbs = Ez->getTotalPosition (pos);

        FieldPointValue* valEz = Ez->getFieldPointValue (pos);

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::EZ, Eps, GridType::EPS);

        GridCoordinate3D posLeft = yeeLayout->getEzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getEzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout->getEzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getEzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valHy1 = Hy->getFieldPointValue (posRight);
        FieldPointValue* valHy2 = Hy->getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx->getFieldPointValue (posUp);
        FieldPointValue* valHx2 = Hx->getFieldPointValue (posDown);

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();
        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateEzTFSF (posAbs, prevHy1, prevHy2, prevHx1, prevHx2, posLeft, posRight, posDown, posUp);
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
        GridCoordinate3D posAbs = Ez->getTotalPosition (pos);

        FieldPointValue* valDz = Dz->getFieldPointValue (pos);

        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::DZ, SigmaX, GridType::SIGMAX);

        GridCoordinate3D posLeft = yeeLayout->getEzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getEzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout->getEzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getEzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valHy1 = Hy->getFieldPointValue (posRight);
        FieldPointValue* valHy2 = Hy->getFieldPointValue (posLeft);

        FieldPointValue* valHx1 = Hx->getFieldPointValue (posUp);
        FieldPointValue* valHx2 = Hx->getFieldPointValue (posDown);

        FieldValue prevHx1 = valHx1->getPrevValue ();
        FieldValue prevHx2 = valHx2->getPrevValue ();
        FieldValue prevHy1 = valHy1->getPrevValue ();
        FieldValue prevHy2 = valHy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateEzTFSF (posAbs, prevHy1, prevHy2, prevHx1, prevHx2, posLeft, posRight, posDown, posUp);
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
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Ez->getTotalPosition (pos);

          FieldPointValue* valD1z = D1z->getFieldPointValue (pos);
          FieldPointValue* valDz = Dz->getFieldPointValue (pos);

          FPValue omegaPE;
          FPValue gammaE;
          FPValue eps = yeeLayout->getMetaMaterial (posAbs, GridType::DZ, Eps, GridType::EPS, OmegaPE, GridType::OMEGAPE, GammaE, GridType::GAMMAE, omegaPE, gammaE);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
  {
    for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
    {
      for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Ez->getTotalPosition (pos);

        FieldPointValue* valEz = Ez->getFieldPointValue (pos);
        FieldPointValue* valDz;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valDz = D1z->getFieldPointValue (pos);
        }
        else
        {
          valDz = Dz->getFieldPointValue (pos);
        }

        FPValue eps = yeeLayout->getMaterial (posAbs, GridType::DZ, Eps, GridType::EPS);
        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::DZ, SigmaY, GridType::SIGMAY);
        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::DZ, SigmaZ, GridType::SIGMAZ);

        FPValue modifier = eps * eps0;
        if (solverSettings.getDoUseMetamaterials ())
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
}

void
Scheme3D::performHxSteps (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (solverSettings.getDoUsePML ())
  {
    calculateHxStepPML (t, HxStart, HxEnd);
  }
  else
  {
    calculateHxStep (t, HxStart, HxEnd);
  }

  if (solverSettings.getDoUsePointSourceHx ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HX)> (t);
  }
}

void
Scheme3D::calculateHxTFSF (GridCoordinate3D posAbs,
                           FieldValue &valEz1,
                           FieldValue &valEz2,
                           FieldValue &valEy1,
                           FieldValue &valEy2,
                           GridCoordinate3D posDown,
                           GridCoordinate3D posUp,
                           GridCoordinate3D posBack,
                           GridCoordinate3D posFront)
{
  bool do_need_update_down = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN, DO_USE_3D_MODE);
  bool do_need_update_up = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP, DO_USE_3D_MODE);

  bool do_need_update_back = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::BACK, DO_USE_3D_MODE);
  bool do_need_update_front = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::FRONT, DO_USE_3D_MODE);

  GridCoordinate3D auxPosY;
  GridCoordinate3D auxPosZ;
  FieldValue diffY;
  FieldValue diffZ;

  if (do_need_update_down)
  {
    auxPosY = posDown;
  }
  else if (do_need_update_up)
  {
    auxPosY = posUp;
  }

  if (do_need_update_back)
  {
    auxPosZ = posBack;
  }
  else if (do_need_update_front)
  {
    auxPosZ = posFront;
  }

  if (do_need_update_down || do_need_update_up)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPosY));

    diffY = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_back || do_need_update_front)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPosZ));

    diffZ = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_down)
  {
    valEz2 += diffY;
  }
  else if (do_need_update_up)
  {
    valEz1 += diffY;
  }

  if (do_need_update_back)
  {
    valEy2 += diffZ;
  }
  else if (do_need_update_front)
  {
    valEy1 += diffZ;
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
        GridCoordinate3D posAbs = Hx->getTotalPosition (pos);

        FieldPointValue* valHx = Hx->getFieldPointValue (pos);

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::HX, Mu, GridType::MU);

        GridCoordinate3D posDown = yeeLayout->getHxCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getHxCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout->getHxCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getHxCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEz1 = Ez->getFieldPointValue (posUp);
        FieldPointValue* valEz2 = Ez->getFieldPointValue (posDown);

        FieldPointValue* valEy1 = Ey->getFieldPointValue (posFront);
        FieldPointValue* valEy2 = Ey->getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHxTFSF (posAbs, prevEz1, prevEz2, prevEy1, prevEy2, posDown, posUp, posBack, posFront);
        }

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
        GridCoordinate3D posAbs = Hx->getTotalPosition (pos);

        FieldPointValue* valBx = Bx->getFieldPointValue (pos);

        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::BX, SigmaY, GridType::SIGMAY);

        GridCoordinate3D posDown = yeeLayout->getHxCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getHxCircuitElement (pos, LayoutDirection::UP);
        GridCoordinate3D posBack = yeeLayout->getHxCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getHxCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEz1 = Ez->getFieldPointValue (posUp);
        FieldPointValue* valEz2 = Ez->getFieldPointValue (posDown);

        FieldPointValue* valEy1 = Ey->getFieldPointValue (posFront);
        FieldPointValue* valEy2 = Ey->getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHxTFSF (posAbs, prevEz1, prevEz2, prevEy1, prevEy2, posDown, posUp, posBack, posFront);
        }

        FPValue k_y = 1;

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

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hx->getTotalPosition (pos);

          FieldPointValue* valB1x = B1x->getFieldPointValue (pos);
          FieldPointValue* valBx = Bx->getFieldPointValue (pos);

          FPValue omegaPM;
          FPValue gammaM;
          FPValue mu = yeeLayout->getMetaMaterial (posAbs, GridType::BX, Mu, GridType::MU, OmegaPM, GridType::OMEGAPM, GammaM, GridType::GAMMAM, omegaPM, gammaM);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
  {
    for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
    {
      for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Hx->getTotalPosition (pos);

        FieldPointValue* valHx = Hx->getFieldPointValue (pos);

        FieldPointValue* valBx;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valBx = B1x->getFieldPointValue (pos);
        }
        else
        {
          valBx = Bx->getFieldPointValue (pos);
        }

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::BX, Mu, GridType::MU);
        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::BX, SigmaX, GridType::SIGMAX);
        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::BX, SigmaZ, GridType::SIGMAZ);

        FPValue modifier = mu * mu0;
        if (solverSettings.getDoUseMetamaterials ())
        {
          modifier = 1;
        }

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
}

void
Scheme3D::performHySteps (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (solverSettings.getDoUsePML ())
  {
    calculateHyStepPML (t, HyStart, HyEnd);
  }
  else
  {
    calculateHyStep (t, HyStart, HyEnd);
  }

  if (solverSettings.getDoUsePointSourceHy ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HY)> (t);
  }
}

void
Scheme3D::calculateHyTFSF (GridCoordinate3D posAbs,
                           FieldValue &valEz1,
                           FieldValue &valEz2,
                           FieldValue &valEx1,
                           FieldValue &valEx2,
                           GridCoordinate3D posLeft,
                           GridCoordinate3D posRight,
                           GridCoordinate3D posBack,
                           GridCoordinate3D posFront)
{
  bool do_need_update_left = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT, DO_USE_3D_MODE);
  bool do_need_update_right = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT, DO_USE_3D_MODE);

  bool do_need_update_back = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::BACK, DO_USE_3D_MODE);
  bool do_need_update_front = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::FRONT, DO_USE_3D_MODE);

  GridCoordinate3D auxPosX;
  GridCoordinate3D auxPosZ;
  FieldValue diffX;
  FieldValue diffZ;

  if (do_need_update_left)
  {
    auxPosX = posLeft;
  }
  else if (do_need_update_right)
  {
    auxPosX = posRight;
  }

  if (do_need_update_back)
  {
    auxPosZ = posBack;
  }
  else if (do_need_update_front)
  {
    auxPosZ = posFront;
  }

  if (do_need_update_left || do_need_update_right)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPosX));

    diffX = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_back || do_need_update_front)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPosZ));

    diffZ = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_left)
  {
    valEz2 += diffX;
  }
  else if (do_need_update_right)
  {
    valEz1 += diffX;
  }

  if (do_need_update_back)
  {
    valEx2 += diffZ;
  }
  else if (do_need_update_front)
  {
    valEx1 += diffZ;
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
        GridCoordinate3D posAbs = Hy->getTotalPosition (pos);

        FieldPointValue* valHy = Hy->getFieldPointValue (pos);

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::HY, Mu, GridType::MU);

        GridCoordinate3D posLeft = yeeLayout->getHyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getHyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout->getHyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getHyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEz1 = Ez->getFieldPointValue (posRight);
        FieldPointValue* valEz2 = Ez->getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex->getFieldPointValue (posFront);
        FieldPointValue* valEx2 = Ex->getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHyTFSF (posAbs, prevEz1, prevEz2, prevEx1, prevEx2, posLeft, posRight, posBack, posFront);
        }

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
        GridCoordinate3D posAbs = Hy->getTotalPosition (pos);

        FieldPointValue* valBy = By->getFieldPointValue (pos);

        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::BY, SigmaZ, GridType::SIGMAZ);

        GridCoordinate3D posLeft = yeeLayout->getHyCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getHyCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posBack = yeeLayout->getHyCircuitElement (pos, LayoutDirection::BACK);
        GridCoordinate3D posFront = yeeLayout->getHyCircuitElement (pos, LayoutDirection::FRONT);

        FieldPointValue* valEz1 = Ez->getFieldPointValue (posRight);
        FieldPointValue* valEz2 = Ez->getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex->getFieldPointValue (posFront);
        FieldPointValue* valEx2 = Ex->getFieldPointValue (posBack);

        FieldValue prevEz1 = valEz1->getPrevValue ();
        FieldValue prevEz2 = valEz2->getPrevValue ();

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHyTFSF (posAbs, prevEz1, prevEz2, prevEx1, prevEx2, posLeft, posRight, posBack, posFront);
        }

        FPValue k_z = 1;

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

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hy->getTotalPosition (pos);

          FieldPointValue* valB1y = B1y->getFieldPointValue (pos);
          FieldPointValue* valBy = By->getFieldPointValue (pos);

          FPValue omegaPM;
          FPValue gammaM;
          FPValue mu = yeeLayout->getMetaMaterial (posAbs, GridType::BY, Mu, GridType::MU, OmegaPM, GridType::OMEGAPM, GammaM, GridType::GAMMAM, omegaPM, gammaM);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
  {
    for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
    {
      for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Hy->getTotalPosition (pos);

        FieldPointValue* valHy = Hy->getFieldPointValue (pos);

        FieldPointValue* valBy;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valBy = B1y->getFieldPointValue (pos);
        }
        else
        {
          valBy = By->getFieldPointValue (pos);
        }

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::BY, Mu, GridType::MU);
        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::BY, SigmaX, GridType::SIGMAX);
        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::BY, SigmaY, GridType::SIGMAY);

        FPValue modifier = mu * mu0;
        if (solverSettings.getDoUseMetamaterials ())
        {
          modifier = 1;
        }

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
}

void
Scheme3D::performHzSteps (time_step t, GridCoordinate3D HzStart, GridCoordinate3D HzEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (solverSettings.getDoUsePML ())
  {
    calculateHzStepPML (t, HzStart, HzEnd);
  }
  else
  {
    calculateHzStep (t, HzStart, HzEnd);
  }

  if (solverSettings.getDoUsePointSourceHz ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HZ)> (t);
  }
}

void
Scheme3D::calculateHzTFSF (GridCoordinate3D posAbs,
                           FieldValue &valEx1,
                           FieldValue &valEx2,
                           FieldValue &valEy1,
                           FieldValue &valEy2,
                           GridCoordinate3D posLeft,
                           GridCoordinate3D posRight,
                           GridCoordinate3D posDown,
                           GridCoordinate3D posUp)
{
  bool do_need_update_left = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT, DO_USE_3D_MODE);
  bool do_need_update_right = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT, DO_USE_3D_MODE);
  bool do_need_update_down = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN, DO_USE_3D_MODE);
  bool do_need_update_up = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP, DO_USE_3D_MODE);

  GridCoordinate3D auxPosX;
  GridCoordinate3D auxPosY;
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
    GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPosY));

    diffY = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_left || do_need_update_right)
  {
    GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPosX));

    diffX = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));
  }

  if (do_need_update_down)
  {
    valEx2 += diffY;
  }
  else if (do_need_update_up)
  {
    valEx1 += diffY;
  }

  if (do_need_update_left)
  {
    valEy2 += diffX;
  }
  else if (do_need_update_right)
  {
    valEy1 += diffX;
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
        GridCoordinate3D posAbs = Hz->getTotalPosition (pos);

        FieldPointValue* valHz = Hz->getFieldPointValue (pos);

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::HZ, Mu, GridType::MU);

        GridCoordinate3D posLeft = yeeLayout->getHzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getHzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout->getHzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getHzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valEy1 = Ey->getFieldPointValue (posRight);
        FieldPointValue* valEy2 = Ey->getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex->getFieldPointValue (posUp);
        FieldPointValue* valEx2 = Ex->getFieldPointValue (posDown);

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHzTFSF (posAbs, prevEx1, prevEx2, prevEy1, prevEy2, posLeft, posRight, posDown, posUp);
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
        GridCoordinate3D posAbs = Hz->getTotalPosition (pos);

        FieldPointValue* valBz = Bz->getFieldPointValue (pos);

        FPValue sigmaX = yeeLayout->getMaterial (posAbs, GridType::BZ, SigmaX, GridType::SIGMAX);

        GridCoordinate3D posLeft = yeeLayout->getHzCircuitElement (pos, LayoutDirection::LEFT);
        GridCoordinate3D posRight = yeeLayout->getHzCircuitElement (pos, LayoutDirection::RIGHT);
        GridCoordinate3D posDown = yeeLayout->getHzCircuitElement (pos, LayoutDirection::DOWN);
        GridCoordinate3D posUp = yeeLayout->getHzCircuitElement (pos, LayoutDirection::UP);

        FieldPointValue* valEy1 = Ey->getFieldPointValue (posRight);
        FieldPointValue* valEy2 = Ey->getFieldPointValue (posLeft);

        FieldPointValue* valEx1 = Ex->getFieldPointValue (posUp);
        FieldPointValue* valEx2 = Ex->getFieldPointValue (posDown);

        FieldValue prevEx1 = valEx1->getPrevValue ();
        FieldValue prevEx2 = valEx2->getPrevValue ();

        FieldValue prevEy1 = valEy1->getPrevValue ();
        FieldValue prevEy2 = valEy2->getPrevValue ();

        if (solverSettings.getDoUseTFSF ())
        {
          calculateHzTFSF (posAbs, prevEx1, prevEx2, prevEy1, prevEy2, posLeft, posRight, posDown, posUp);
        }

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
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
#ifdef TWO_TIME_STEPS
    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hz->getTotalPosition (pos);

          FieldPointValue* valB1z = B1z->getFieldPointValue (pos);
          FieldPointValue* valBz = Bz->getFieldPointValue (pos);

          FPValue omegaPM;
          FPValue gammaM;
          FPValue mu = yeeLayout->getMetaMaterial (posAbs, GridType::BZ, Mu, GridType::MU, OmegaPM, GridType::OMEGAPM, GammaM, GridType::GAMMAM, omegaPM, gammaM);

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
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
  }

  for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
  {
    for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
    {
      for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);
        GridCoordinate3D posAbs = Hz->getTotalPosition (pos);

        FieldPointValue* valHz = Hz->getFieldPointValue (pos);

        FieldPointValue* valBz;

        if (solverSettings.getDoUseMetamaterials ())
        {
          valBz = B1z->getFieldPointValue (pos);
        }
        else
        {
          valBz = Bz->getFieldPointValue (pos);
        }

        FPValue mu = yeeLayout->getMaterial (posAbs, GridType::BZ, Mu, GridType::MU);
        FPValue sigmaY = yeeLayout->getMaterial (posAbs, GridType::BZ, SigmaY, GridType::SIGMAY);
        FPValue sigmaZ = yeeLayout->getMaterial (posAbs, GridType::BZ, SigmaZ, GridType::SIGMAZ);

        FPValue modifier = mu * mu0;
        if (solverSettings.getDoUseMetamaterials ())
        {
          modifier = 1;
        }

        FPValue k_y = 1;
        FPValue k_z = 1;

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
Scheme3D::performNSteps (time_step startStep, time_step numberTimeSteps)
{
  time_step diffT = solverSettings.getRebalanceStep ();

  int processId = 0;

  time_step stepLimit = startStep + numberTimeSteps;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (processId == 0)
  {
    DPRINTF (LOG_LEVEL_STAGES, "Performing computations for [%u,%u] time steps.\n", startStep, stepLimit);
  }

  for (time_step t = startStep; t < stepLimit; ++t)
  {
    if (processId == 0)
    {
      DPRINTF (LOG_LEVEL_STAGES, "Calculating time step %u...\n", t);
    }

    GridCoordinate3D ExStart = Ex->getComputationStart (yeeLayout->getExStartDiff ());
    GridCoordinate3D ExEnd = Ex->getComputationEnd (yeeLayout->getExEndDiff ());

    GridCoordinate3D EyStart = Ey->getComputationStart (yeeLayout->getEyStartDiff ());
    GridCoordinate3D EyEnd = Ey->getComputationEnd (yeeLayout->getEyEndDiff ());

    GridCoordinate3D EzStart = Ez->getComputationStart (yeeLayout->getEzStartDiff ());
    GridCoordinate3D EzEnd = Ez->getComputationEnd (yeeLayout->getEzEndDiff ());

    GridCoordinate3D HxStart = Hx->getComputationStart (yeeLayout->getHxStartDiff ());
    GridCoordinate3D HxEnd = Hx->getComputationEnd (yeeLayout->getHxEndDiff ());

    GridCoordinate3D HyStart = Hy->getComputationStart (yeeLayout->getHyStartDiff ());
    GridCoordinate3D HyEnd = Hy->getComputationEnd (yeeLayout->getHyEndDiff ());

    GridCoordinate3D HzStart = Hz->getComputationStart (yeeLayout->getHzStartDiff ());
    GridCoordinate3D HzEnd = Hz->getComputationEnd (yeeLayout->getHzEndDiff ());

    if (solverSettings.getDoUseParallelGrid () && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
    }

    if (solverSettings.getDoUseTFSF ())
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);
    performEzSteps (t, EzStart, EzEnd);

    if (solverSettings.getDoUseParallelGrid () && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
    }

    Ex->nextTimeStep ();
    Ey->nextTimeStep ();
    Ez->nextTimeStep ();

    if (solverSettings.getDoUsePML ())
    {
      Dx->nextTimeStep ();
      Dy->nextTimeStep ();
      Dz->nextTimeStep ();
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      D1x->nextTimeStep ();
      D1y->nextTimeStep ();
      D1z->nextTimeStep ();
    }

    if (solverSettings.getDoUseParallelGrid () && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
    }

    if (solverSettings.getDoUseTFSF ())
    {
      performPlaneWaveHSteps (t);
    }

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);
    performHzSteps (t, HzStart, HzEnd);

    if (solverSettings.getDoUseParallelGrid () && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
    }

    Hx->nextTimeStep ();
    Hy->nextTimeStep ();
    Hz->nextTimeStep ();

    if (solverSettings.getDoUsePML ())
    {
      Bx->nextTimeStep ();
      By->nextTimeStep ();
      Bz->nextTimeStep ();
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      B1x->nextTimeStep ();
      B1y->nextTimeStep ();
      B1z->nextTimeStep ();
    }

    if (solverSettings.getDoSaveIntermediateRes ()
        && t % solverSettings.getIntermediateSaveStep () == 0)
    {
      gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldIntermediate ());
      saveGrids (t);
    }

    if (solverSettings.getDoUseNTFF ()
        && t % solverSettings.getIntermediateNTFFStep () == 0)
    {
      gatherFieldsTotal (solverSettings.getDoCalcScatteredNTFF ());
      saveNTFF (solverSettings.getDoCalcReverseNTFF (), t);
    }

    additionalUpdateOfGrids (t, diffT);
  }

  if (solverSettings.getDoSaveRes ())
  {
    gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldRes ());
    saveGrids (stepLimit);
  }
}

void
Scheme3D::performAmplitudeSteps (time_step startStep)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

  ASSERT_MESSAGE ("Temporary unsupported");

  int processId = 0;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  int is_stable_state = 0;

  GridCoordinate3D EzSize = Ez->getSize ();

  time_step t = startStep;

  while (is_stable_state == 0 && t < solverSettings.getNumAmplitudeSteps ())
  {
    FPValue maxAccuracy = -1;

    //is_stable_state = 1;

    GridCoordinate3D ExStart = Ex->getComputationStart (yeeLayout->getExStartDiff ());
    GridCoordinate3D ExEnd = Ex->getComputationEnd (yeeLayout->getExEndDiff ());

    GridCoordinate3D EyStart = Ey->getComputationStart (yeeLayout->getEyStartDiff ());
    GridCoordinate3D EyEnd = Ey->getComputationEnd (yeeLayout->getEyEndDiff ());

    GridCoordinate3D EzStart = Ez->getComputationStart (yeeLayout->getEzStartDiff ());
    GridCoordinate3D EzEnd = Ez->getComputationEnd (yeeLayout->getEzEndDiff ());

    GridCoordinate3D HxStart = Hx->getComputationStart (yeeLayout->getHxStartDiff ());
    GridCoordinate3D HxEnd = Hx->getComputationEnd (yeeLayout->getHxEndDiff ());

    GridCoordinate3D HyStart = Hy->getComputationStart (yeeLayout->getHyStartDiff ());
    GridCoordinate3D HyEnd = Hy->getComputationEnd (yeeLayout->getHyEndDiff ());

    GridCoordinate3D HzStart = Hz->getComputationStart (yeeLayout->getHzStartDiff ());
    GridCoordinate3D HzEnd = Hz->getComputationEnd (yeeLayout->getHzEndDiff ());

    if (solverSettings.getDoUseTFSF ())
    {
      performPlaneWaveESteps (t);
    }

    performExSteps (t, ExStart, ExEnd);
    performEySteps (t, EyStart, EyEnd);
    performEzSteps (t, EzStart, EzEnd);

    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);

          if (!yeeLayout->isExInPML (Ex->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ex->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = ExAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

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

          if (!yeeLayout->isEyInPML (Ey->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ey->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = EyAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

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

          if (!yeeLayout->isEzInPML (Ez->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Ez->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = EzAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    Ex->nextTimeStep ();
    Ey->nextTimeStep ();
    Ez->nextTimeStep ();

    if (solverSettings.getDoUsePML ())
    {
      Dx->nextTimeStep ();
      Dy->nextTimeStep ();
      Dz->nextTimeStep ();
    }

    if (solverSettings.getDoUseTFSF ())
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

          if (!yeeLayout->isHxInPML (Hx->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hx->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HxAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

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

          if (!yeeLayout->isHyInPML (Hy->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hy->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HyAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

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

          if (!yeeLayout->isHzInPML (Hz->getTotalPosition (pos)))
          {
            FieldPointValue* tmp = Hz->getFieldPointValue (pos);
            FieldPointValue* tmpAmp = HzAmplitude->getFieldPointValue (pos);

            GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (pos));

            GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
            GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());

            FPValue val = tmp->getCurValue ();

            if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
            {
              is_stable_state = 0;
            }
          }
        }
      }
    }

    Hx->nextTimeStep ();
    Hy->nextTimeStep ();
    Hz->nextTimeStep ();

    if (solverSettings.getDoUsePML ())
    {
      Bx->nextTimeStep ();
      By->nextTimeStep ();
      Bz->nextTimeStep ();
    }

    ++t;

    if (maxAccuracy < 0)
    {
      is_stable_state = 0;
    }

    DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " PRINTF_MODIFIER ". \n", t, maxAccuracy);
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.");
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
Scheme3D::performSteps ()
{
#if defined (CUDA_ENABLED)

  int processId = 0;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (solverSettings.getDoUsePML ()
      || solverSettings.getDoUseTFSF ()
      || solverSettings.getDoUseAmplitudeMode ()
      || solverSettings.getDoUseMetamaterials ())
  {
    ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");
  }

  CudaExitStatus status;

  cudaExecute3DSteps (&status, yeeLayout, gridTimeStep, gridStep, Ex, Ey, Ez, Hx, Hy, Hz, Eps, Mu, totalStep, processId);

  ASSERT (status == CUDA_OK);

  if (solverSettings.getDoSaveRes ())
  {
    gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldRes ());
    saveGrids (totalStep);
  }

#else /* CUDA_ENABLED */

  if (solverSettings.getDoUseMetamaterials () && !solverSettings.getDoUsePML ())
  {
    ASSERT_MESSAGE ("Metamaterials without pml are not implemented");
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    if (solverSettings.getDoUseAmplitudeMode ())
    {
      ASSERT_MESSAGE ("Parallel amplitude mode is not implemented");
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  performNSteps (0, totalStep);

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    performAmplitudeSteps (totalStep);
  }

#endif /* !CUDA_ENABLED */
}

void
Scheme3D::initScheme (FPValue dx, FPValue sourceFreq)
{
  sourceFrequency = sourceFreq;
  sourceWaveLength = PhysicsConst::SpeedOfLight / sourceFrequency;

  gridStep = dx;
  courantNum = 1.0 / 2.0;
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  FPValue phaseVelocity0 = Approximation::phaseVelocityIncidentWave3D (gridStep, sourceWaveLength, courantNum, N_lambda, PhysicsConst::Pi / 2, 0);
  FPValue phaseVelocity = Approximation::phaseVelocityIncidentWave3D (gridStep, sourceWaveLength, courantNum, N_lambda, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ());

  relPhaseVelocity = phaseVelocity0 / phaseVelocity;
}

void
Scheme3D::initGrids ()
{
  int processId = 0;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    if (!solverSettings.getEpsFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalEps->initialize ();
    }
    if (!solverSettings.getMuFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalMu->initialize ();
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (!solverSettings.getOmegaPEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPE->initialize ();
      }
      if (!solverSettings.getOmegaPMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPM->initialize ();
      }
      if (!solverSettings.getGammaEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaE->initialize ();
      }
      if (!solverSettings.getGammaMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaM->initialize ();
      }
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  Eps->initialize (getFieldValueRealOnly (1.0));

  if (!solverSettings.getEpsFileName ().empty ())
  {
    FileType type = GridFileManager::getFileType (solverSettings.getEpsFileName ());
    loader[type]->initManual (0, CURRENT, processId, solverSettings.getEpsFileName (), "", "");
    loader[type]->loadGrid (totalEps);

    if (solverSettings.getDoUseParallelGrid ())
    {
      for (int i = 0; i < Eps->getSize ().getX (); ++i)
      {
        for (int j = 0; j < Eps->getSize ().getY (); ++j)
        {
          for (int k = 0; k < Eps->getSize ().getZ (); ++k)
          {
            GridCoordinate3D pos (i, j, k);
            GridCoordinate3D posAbs = Eps->getTotalPosition (pos);
            FieldPointValue *val = Eps->getFieldPointValue (pos);
            *val = *totalEps->getFieldPointValue (posAbs);
          }
        }
      }
    }
  }

  if (solverSettings.getEpsSphere () != 1)
  {
    for (int i = 0; i < Eps->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Eps->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Eps->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinateFP3D posAbs = yeeLayout->getEpsCoordFP (Eps->getTotalPosition (pos));
          FieldPointValue *val = Eps->getFieldPointValue (pos);

          FieldValue epsVal = getFieldValueRealOnly (solverSettings.getEpsSphere ());

          FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);
          GridCoordinateFP3D center (solverSettings.getEpsSphereCenterX (),
                                     solverSettings.getEpsSphereCenterY (),
                                     solverSettings.getEpsSphereCenterZ ());
          val->setCurValue (Approximation::approximateSphere (posAbs,
                                                              center * modifier + GridCoordinateFP3D (0.5, 0.5, 0.5),
                                                              solverSettings.getEpsSphereRadius () * modifier, epsVal));
        }
      }
    }
  }

  Mu->initialize (getFieldValueRealOnly (1.0));

  if (!solverSettings.getMuFileName ().empty ())
  {
    FileType type = GridFileManager::getFileType (solverSettings.getMuFileName ());
    loader[type]->initManual (0, CURRENT, processId, solverSettings.getMuFileName (), "", "");
    loader[type]->loadGrid (totalMu);

    if (solverSettings.getDoUseParallelGrid ())
    {
      for (int i = 0; i < Mu->getSize ().getX (); ++i)
      {
        for (int j = 0; j < Mu->getSize ().getY (); ++j)
        {
          for (int k = 0; k < Mu->getSize ().getZ (); ++k)
          {
            GridCoordinate3D pos (i, j, k);
            GridCoordinate3D posAbs = Mu->getTotalPosition (pos);
            FieldPointValue *val = Mu->getFieldPointValue (pos);
            *val = *totalMu->getFieldPointValue (posAbs);
          }
        }
      }
    }
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
    OmegaPE->initialize ();

    if (!solverSettings.getOmegaPEFileName ().empty ())
    {
      FileType type = GridFileManager::getFileType (solverSettings.getOmegaPEFileName ());
      loader[type]->initManual (0, CURRENT, processId, solverSettings.getOmegaPEFileName (), "", "");
      loader[type]->loadGrid (totalOmegaPE);

      if (solverSettings.getDoUseParallelGrid ())
      {
        for (int i = 0; i < OmegaPE->getSize ().getX (); ++i)
        {
          for (int j = 0; j < OmegaPE->getSize ().getY (); ++j)
          {
            for (int k = 0; k < OmegaPE->getSize ().getZ (); ++k)
            {
              GridCoordinate3D pos (i, j, k);
              GridCoordinate3D posAbs = OmegaPE->getTotalPosition (pos);
              FieldPointValue *val = OmegaPE->getFieldPointValue (pos);
              *val = *totalOmegaPE->getFieldPointValue (posAbs);
            }
          }
        }
      }
    }

    OmegaPM->initialize ();

    if (!solverSettings.getOmegaPMFileName ().empty ())
    {
      FileType type = GridFileManager::getFileType (solverSettings.getOmegaPMFileName ());
      loader[type]->initManual (0, CURRENT, processId, solverSettings.getOmegaPMFileName (), "", "");
      loader[type]->loadGrid (totalOmegaPM);

      if (solverSettings.getDoUseParallelGrid ())
      {
        for (int i = 0; i < OmegaPM->getSize ().getX (); ++i)
        {
          for (int j = 0; j < OmegaPM->getSize ().getY (); ++j)
          {
            for (int k = 0; k < OmegaPM->getSize ().getZ (); ++k)
            {
              GridCoordinate3D pos (i, j, k);
              GridCoordinate3D posAbs = OmegaPM->getTotalPosition (pos);
              FieldPointValue *val = OmegaPM->getFieldPointValue (pos);
              *val = *totalOmegaPM->getFieldPointValue (posAbs);
            }
          }
        }
      }
    }

    GammaE->initialize ();

    if (!solverSettings.getGammaEFileName ().empty ())
    {
      FileType type = GridFileManager::getFileType (solverSettings.getGammaEFileName ());
      loader[type]->initManual (0, CURRENT, processId, solverSettings.getGammaEFileName (), "", "");
      loader[type]->loadGrid (totalGammaE);

      if (solverSettings.getDoUseParallelGrid ())
      {
        for (int i = 0; i < GammaE->getSize ().getX (); ++i)
        {
          for (int j = 0; j < GammaE->getSize ().getY (); ++j)
          {
            for (int k = 0; k < GammaE->getSize ().getZ (); ++k)
            {
              GridCoordinate3D pos (i, j, k);
              GridCoordinate3D posAbs = GammaE->getTotalPosition (pos);
              FieldPointValue *val = GammaE->getFieldPointValue (pos);
              *val = *totalGammaE->getFieldPointValue (posAbs);
            }
          }
        }
      }
    }

    GammaM->initialize ();

    if (!solverSettings.getGammaMFileName ().empty ())
    {
      FileType type = GridFileManager::getFileType (solverSettings.getGammaMFileName ());
      loader[type]->initManual (0, CURRENT, processId, solverSettings.getGammaMFileName (), "", "");
      loader[type]->loadGrid (totalGammaM);

      if (solverSettings.getDoUseParallelGrid ())
      {
        for (int i = 0; i < GammaM->getSize ().getX (); ++i)
        {
          for (int j = 0; j < GammaM->getSize ().getY (); ++j)
          {
            for (int k = 0; k < GammaM->getSize ().getZ (); ++k)
            {
              GridCoordinate3D pos (i, j, k);
              GridCoordinate3D posAbs = GammaM->getTotalPosition (pos);
              FieldPointValue *val = GammaM->getFieldPointValue (pos);
              *val = *totalGammaM->getFieldPointValue (posAbs);
            }
          }
        }
      }
    }
  }

  if (solverSettings.getDoUsePML ())
  {
    FPValue eps0 = PhysicsConst::Eps0;
    FPValue mu0 = PhysicsConst::Mu0;

    GridCoordinate3D PMLSize = yeeLayout->getLeftBorderPML () * (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

    FPValue boundary = PMLSize.getX () * gridStep;
    uint32_t exponent = 6;
  	FPValue R_err = 1e-16;
  	FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
  	FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

    for (int i = 0; i < SigmaX->getSize ().getX (); ++i)
    {
      for (int j = 0; j < SigmaX->getSize ().getY (); ++j)
      {
        for (int k = 0; k < SigmaX->getSize ().getZ (); ++k)
        {
          FieldPointValue* valSigma = new FieldPointValue ();

          GridCoordinate3D pos (i, j, k);
          GridCoordinateFP3D posAbs = yeeLayout->getEpsCoordFP (SigmaX->getTotalPosition (pos));

          GridCoordinateFP3D size = yeeLayout->getEpsCoordFP (SigmaX->getTotalSize ());

          /*
           * FIXME: add layout coordinates for material: sigma, eps, etc.
           */
          if (posAbs.getX () < PMLSize.getX ())
          {
            grid_coord dist = PMLSize.getX () - posAbs.getX ();
      			FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
      			FPValue x2 = dist * gridStep;       // lower bounds for point i

            FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));    //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }
          else if (posAbs.getX () >= size.getX () - PMLSize.getX ())
          {
            grid_coord dist = posAbs.getX () - (size.getX () - PMLSize.getX ());
      			FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
      			FPValue x2 = dist * gridStep;       // lower bounds for point i

      			//std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
      			FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }

          SigmaX->setFieldPointValue (valSigma, pos);
        }
      }
    }

    for (int i = 0; i < SigmaY->getSize ().getX (); ++i)
    {
      for (int j = 0; j < SigmaY->getSize ().getY (); ++j)
      {
        for (int k = 0; k < SigmaY->getSize ().getZ (); ++k)
        {
          FieldPointValue* valSigma = new FieldPointValue ();

          GridCoordinate3D pos (i, j, k);
          GridCoordinateFP3D posAbs = yeeLayout->getEpsCoordFP (SigmaY->getTotalPosition (pos));

          GridCoordinateFP3D size = yeeLayout->getEpsCoordFP (SigmaY->getTotalSize ());

          /*
           * FIXME: add layout coordinates for material: sigma, eps, etc.
           */
          if (posAbs.getY () < PMLSize.getY ())
          {
            grid_coord dist = PMLSize.getY () - posAbs.getY ();
            FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
            FPValue x2 = dist * gridStep;       // lower bounds for point i

            FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }
          else if (posAbs.getY () >= size.getY () - PMLSize.getY ())
          {
            grid_coord dist = posAbs.getY () - (size.getY () - PMLSize.getY ());
            FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
            FPValue x2 = dist * gridStep;       // lower bounds for point i

            //std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
            FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }

          SigmaY->setFieldPointValue (valSigma, pos);
        }
      }
    }

    for (int i = 0; i < SigmaZ->getSize ().getX (); ++i)
    {
      for (int j = 0; j < SigmaZ->getSize ().getY (); ++j)
      {
        for (int k = 0; k < SigmaZ->getSize ().getZ (); ++k)
        {
          FieldPointValue* valSigma = new FieldPointValue ();

          GridCoordinate3D pos (i, j, k);
          GridCoordinateFP3D posAbs = yeeLayout->getEpsCoordFP (SigmaZ->getTotalPosition (pos));

          GridCoordinateFP3D size = yeeLayout->getEpsCoordFP (SigmaZ->getTotalSize ());

          /*
           * FIXME: add layout coordinates for material: sigma, eps, etc.
           */
          if (posAbs.getZ () < PMLSize.getZ ())
          {
            grid_coord dist = PMLSize.getZ () - posAbs.getZ ();
            FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
            FPValue x2 = dist * gridStep;       // lower bounds for point i

            FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }
          else if (posAbs.getZ () >= size.getZ () - PMLSize.getZ ())
          {
            grid_coord dist = posAbs.getZ () - (size.getZ () - PMLSize.getZ ());
            FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
            FPValue x2 = dist * gridStep;       // lower bounds for point i

            //std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
            FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

            valSigma->setCurValue (getFieldValueRealOnly (val));
          }

          SigmaZ->setFieldPointValue (valSigma, pos);
        }
      }
    }
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (solverSettings.getDoSaveMaterials ())
    {
      if (solverSettings.getDoUseParallelGrid ())
      {
#ifdef PARALLEL_GRID
        ((ParallelGrid *) Eps)->gatherFullGridPlacement (totalEps);
        ((ParallelGrid *) Mu)->gatherFullGridPlacement (totalMu);

        if (solverSettings.getDoUseMetamaterials ())
        {
          ((ParallelGrid *) OmegaPE)->gatherFullGridPlacement (totalOmegaPE);
          ((ParallelGrid *) OmegaPM)->gatherFullGridPlacement (totalOmegaPM);
          ((ParallelGrid *) GammaE)->gatherFullGridPlacement (totalGammaE);
          ((ParallelGrid *) GammaM)->gatherFullGridPlacement (totalGammaM);
        }
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
      }

      if (processId == 0)
      {
        dumper[type]->init (0, CURRENT, processId, "Eps");
        dumper[type]->dumpGrid (totalEps,
                                GridCoordinate3D (0, 0, totalEps->getSize ().getZ () / 2),
                                GridCoordinate3D (totalEps->getSize ().getX (), totalEps->getSize ().getY (), totalEps->getSize ().getZ () / 2 + 1));

        dumper[type]->init (0, CURRENT, processId, "Mu");
        dumper[type]->dumpGrid (totalMu,
                                GridCoordinate3D (0, 0, totalMu->getSize ().getZ () / 2),
                                GridCoordinate3D (totalMu->getSize ().getX (), totalMu->getSize ().getY (), totalMu->getSize ().getZ () / 2 + 1));

        if (solverSettings.getDoUseMetamaterials ())
        {
          dumper[type]->init (0, CURRENT, processId, "OmegaPE");
          dumper[type]->dumpGrid (totalOmegaPE,
                                  GridCoordinate3D (0, 0, totalOmegaPE->getSize ().getZ () / 2),
                                  GridCoordinate3D (totalOmegaPE->getSize ().getX (), totalOmegaPE->getSize ().getY (), totalOmegaPE->getSize ().getZ () / 2 + 1));

          dumper[type]->init (0, CURRENT, processId, "OmegaPM");
          dumper[type]->dumpGrid (totalOmegaPM,
                                  GridCoordinate3D (0, 0, totalOmegaPM->getSize ().getZ () / 2),
                                  GridCoordinate3D (totalOmegaPM->getSize ().getX (), totalOmegaPM->getSize ().getY (), totalOmegaPM->getSize ().getZ () / 2 + 1));

          dumper[type]->init (0, CURRENT, processId, "GammaE");
          dumper[type]->dumpGrid (totalGammaE,
                                  GridCoordinate3D (0, 0, totalGammaE->getSize ().getZ () / 2),
                                  GridCoordinate3D (totalGammaE->getSize ().getX (), totalGammaE->getSize ().getY (), totalGammaE->getSize ().getZ () / 2 + 1));

          dumper[type]->init (0, CURRENT, processId, "GammaM");
          dumper[type]->dumpGrid (totalGammaM,
                                  GridCoordinate3D (0, 0, totalGammaM->getSize ().getZ () / 2),
                                  GridCoordinate3D (totalGammaM->getSize ().getX (), totalGammaM->getSize ().getY (), totalGammaM->getSize ().getZ () / 2 + 1));
        }
        //
        // if (solverSettings.getDoUsePML ())
        // {
        //   dumper[type]->init (0, CURRENT, processId, "SigmaX");
        //   dumper[type]->dumpGrid (SigmaX,
        //                           GridCoordinate3D (0, 0, SigmaX->getSize ().getZ () / 2),
        //                           GridCoordinate3D (SigmaX->getSize ().getX (), SigmaX->getSize ().getY (), SigmaX->getSize ().getZ () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "SigmaY");
        //   dumper[type]->dumpGrid (SigmaY,
        //                           GridCoordinate3D (0, 0, SigmaY->getSize ().getZ () / 2),
        //                           GridCoordinate3D (SigmaY->getSize ().getX (), SigmaY->getSize ().getY (), SigmaY->getSize ().getZ () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "SigmaZ");
        //   dumper[type]->dumpGrid (SigmaZ,
        //                           GridCoordinate3D (0, 0, SigmaZ->getSize ().getZ () / 2),
        //                           GridCoordinate3D (SigmaZ->getSize ().getX (), SigmaZ->getSize ().getY (), SigmaZ->getSize ().getZ () / 2 + 1));
        // }
      }
    }
  }

  Ex->initialize ();
  Ey->initialize ();
  Ez->initialize ();
  Hx->initialize ();
  Hy->initialize ();
  Hz->initialize ();

  if (solverSettings.getDoUsePML ())
  {
    Dx->initialize ();
    Dy->initialize ();
    Dz->initialize ();

    Bx->initialize ();
    By->initialize ();
    Bz->initialize ();

    if (solverSettings.getDoUseMetamaterials ())
    {
      D1x->initialize ();
      D1y->initialize ();
      D1z->initialize ();

      B1x->initialize ();
      B1y->initialize ();
      B1z->initialize ();
    }
  }

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    ExAmplitude->initialize ();
    EyAmplitude->initialize ();
    EzAmplitude->initialize ();

    HxAmplitude->initialize ();
    HyAmplitude->initialize ();
    HzAmplitude->initialize ();
  }

  if (solverSettings.getDoUseTFSF ())
  {
    EInc->initialize ();
    HInc->initialize ();
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    MPI_Barrier (MPI_COMM_WORLD);

    ((ParallelGrid *) Eps)->share ();
    ((ParallelGrid *) Mu)->share ();

    if (solverSettings.getDoUsePML ())
    {
      ((ParallelGrid *) SigmaX)->share ();
      ((ParallelGrid *) SigmaY)->share ();
      ((ParallelGrid *) SigmaZ)->share ();
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

Scheme3D::NPair
Scheme3D::ntffN_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEz,
                   Grid<GridCoordinate3D> *curTotalHy,
                   Grid<GridCoordinate3D> *curTotalHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (x0, leftNTFF.getY () + 0.5, leftNTFF.getZ () + 0.5);
  GridCoordinateFP3D coordEnd (x0, rightNTFF.getY () - 0.5, rightNTFF.getZ () - 0.5);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.getY (); coordY <= coordEnd.getY (); ++coordY)
  {
    for (FPValue coordZ = coordStart.getZ (); coordZ <= coordEnd.getZ (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ);
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ);
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5);
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5);

      // FieldValue val1 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos1));
      // FieldValue val2 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos2));
      // FieldValue val3 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos3));
      // FieldValue val4 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos4));

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

      FieldValue valHz1 = curTotalHz->getFieldPointValue (convertCoord (pos1))->getCurValue ();// - val1;
      FieldValue valHz2 = curTotalHz->getFieldPointValue (convertCoord (pos2))->getCurValue ();// - val2;

      FieldValue valHy1 = curTotalHy->getFieldPointValue (convertCoord (pos3))->getCurValue ();// - val3;
      FieldValue valHy2 = curTotalHy->getFieldPointValue (convertCoord (pos4))->getCurValue ();// - val4;

      FPValue arg = (x0 - diffc) * sin(angleTeta)*cos(anglePhi) + (coordY - diffc) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((valHz1 + valHz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (valHy1 + valHy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((valHz1 + valHz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEz,
                   Grid<GridCoordinate3D> *curTotalHx,
                   Grid<GridCoordinate3D> *curTotalHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (leftNTFF.getX () + 0.5, y0, leftNTFF.getZ () + 0.5);
  GridCoordinateFP3D coordEnd (rightNTFF.getX () - 0.5, y0, rightNTFF.getZ () - 0.5);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.getX (); coordX <= coordEnd.getX (); ++coordX)
  {
    for (FPValue coordZ = coordStart.getZ (); coordZ <= coordEnd.getZ (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ);
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ);
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5);
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5);

      // FieldValue val1 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos1));
      // FieldValue val2 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos2));
      // FieldValue val3 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos3));
      // FieldValue val4 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos4));

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      FieldValue valHz1 = curTotalHz->getFieldPointValue (convertCoord (pos1))->getCurValue ();// - val1;
      FieldValue valHz2 = curTotalHz->getFieldPointValue (convertCoord (pos2))->getCurValue ();// - val2;

      FieldValue valHx1 = curTotalHx->getFieldPointValue (convertCoord (pos3))->getCurValue ();// - val3;
      FieldValue valHx2 = curTotalHx->getFieldPointValue (convertCoord (pos4))->getCurValue ();// - val4;

      FPValue arg = (coordX - diffc) * sin(angleTeta)*cos(anglePhi) + (y0 - diffc) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.getY ()?1:-1) * ((valHz1 + valHz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (valHx1 + valHx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.getY ()?1:-1) * ((valHz1 + valHz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEz,
                   Grid<GridCoordinate3D> *curTotalHx,
                   Grid<GridCoordinate3D> *curTotalHy)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (leftNTFF.getX () + 0.5, leftNTFF.getY () + 0.5, z0);
  GridCoordinateFP3D coordEnd (rightNTFF.getX () - 0.5, rightNTFF.getY () - 0.5, z0);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.getX (); coordX <= coordEnd.getX (); ++coordX)
  {
    for (FPValue coordY = coordStart.getY (); coordY <= coordEnd.getY (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0);
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0);
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0);
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0);

      // FieldValue val1 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos1));
      // FieldValue val2 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos2));
      // FieldValue val3 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos3));
      // FieldValue val4 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos4));

      pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      FieldValue valHy1 = curTotalHy->getFieldPointValue (convertCoord (pos1))->getCurValue ();// - val1;
      FieldValue valHy2 = curTotalHy->getFieldPointValue (convertCoord (pos2))->getCurValue ();// - val2;

      FieldValue valHx1 = curTotalHx->getFieldPointValue (convertCoord (pos3))->getCurValue ();// - val3;
      FieldValue valHx2 = curTotalHx->getFieldPointValue (convertCoord (pos4))->getCurValue ();// - val4;

      FPValue arg = (coordX - diffc) * sin(angleTeta)*cos(anglePhi) + (coordY - diffc) * sin(angleTeta)*sin(anglePhi) + (z0 - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * (-(valHy1 + valHy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (valHx1 + valHx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * ((valHy1 + valHy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (valHx1 + valHx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEy,
                   Grid<GridCoordinate3D> *curTotalEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (x0, leftNTFF.getY () + 0.5, leftNTFF.getZ () + 0.5);
  GridCoordinateFP3D coordEnd (x0, rightNTFF.getY () - 0.5, rightNTFF.getZ () - 0.5);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.getY (); coordY <= coordEnd.getY (); ++coordY)
  {
    for (FPValue coordZ = coordStart.getZ (); coordZ <= coordEnd.getZ (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ);
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ);
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5);
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5);

      // FieldValue val1 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos1));
      // FieldValue val2 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos2));
      // FieldValue val3 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos3));
      // FieldValue val4 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos4));

      pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      FieldValue valEy1 = (curTotalEy->getFieldPointValue (convertCoord (pos1-GridCoordinateFP3D(0.5,0,0)))->getCurValue ()
                           + curTotalEy->getFieldPointValue (convertCoord (pos1+GridCoordinateFP3D(0.5,0,0)))->getCurValue ()) / FPValue(2.0);// - val1;
      FieldValue valEy2 = (curTotalEy->getFieldPointValue (convertCoord (pos2-GridCoordinateFP3D(0.5,0,0)))->getCurValue ()
                           + curTotalEy->getFieldPointValue (convertCoord (pos2+GridCoordinateFP3D(0.5,0,0)))->getCurValue ()) / FPValue(2.0);// - val2;

      FieldValue valEz1 = (curTotalEz->getFieldPointValue (convertCoord (pos3-GridCoordinateFP3D(0.5,0,0)))->getCurValue ()
                           + curTotalEz->getFieldPointValue (convertCoord (pos3+GridCoordinateFP3D(0.5,0,0)))->getCurValue ()) / FPValue(2.0);// - val3;
      FieldValue valEz2 = (curTotalEz->getFieldPointValue (convertCoord (pos4-GridCoordinateFP3D(0.5,0,0)))->getCurValue ()
                           + curTotalEz->getFieldPointValue (convertCoord (pos4+GridCoordinateFP3D(0.5,0,0)))->getCurValue ()) / FPValue(2.0);// - val4;

      FPValue arg = (x0 - diffc) * sin(angleTeta)*cos(anglePhi) + (coordY - diffc) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((valEz1 + valEz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (valEy1 + valEy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((valEz1 + valEz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEx,
                   Grid<GridCoordinate3D> *curTotalEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (leftNTFF.getX () + 0.5, y0, leftNTFF.getZ () + 0.5);
  GridCoordinateFP3D coordEnd (rightNTFF.getX () - 0.5, y0, rightNTFF.getZ () - 0.5);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.getX (); coordX <= coordEnd.getX (); ++coordX)
  {
    for (FPValue coordZ = coordStart.getZ (); coordZ <= coordEnd.getZ (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ);
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ);
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5);
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5);

      // FieldValue val1 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos1));
      // FieldValue val2 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos2));
      // FieldValue val3 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos3));
      // FieldValue val4 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos4));

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      FieldValue valEx1 = (curTotalEx->getFieldPointValue (convertCoord (pos1-GridCoordinateFP3D(0,0.5,0)))->getCurValue ()
                           + curTotalEx->getFieldPointValue (convertCoord (pos1+GridCoordinateFP3D(0,0.5,0)))->getCurValue ()) / FPValue(2.0);// - val1;
      FieldValue valEx2 = (curTotalEx->getFieldPointValue (convertCoord (pos2-GridCoordinateFP3D(0,0.5,0)))->getCurValue ()
                           + curTotalEx->getFieldPointValue (convertCoord (pos2+GridCoordinateFP3D(0,0.5,0)))->getCurValue ()) / FPValue(2.0);// - val2;

      FieldValue valEz1 = (curTotalEz->getFieldPointValue (convertCoord (pos3-GridCoordinateFP3D(0,0.5,0)))->getCurValue ()
                           + curTotalEz->getFieldPointValue (convertCoord (pos3+GridCoordinateFP3D(0,0.5,0)))->getCurValue ()) / FPValue(2.0);// - val3;
      FieldValue valEz2 = (curTotalEz->getFieldPointValue (convertCoord (pos4-GridCoordinateFP3D(0,0.5,0)))->getCurValue ()
                           + curTotalEz->getFieldPointValue (convertCoord (pos4+GridCoordinateFP3D(0,0.5,0)))->getCurValue ()) / FPValue(2.0);// - val4;

      FPValue arg = (coordX - diffc) * sin(angleTeta)*cos(anglePhi) + (y0 - diffc) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.getY ()?1:-1) * ((valEz1 + valEz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (valEx1 + valEx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.getY ()?1:-1) * ((valEz1 + valEz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curTotalEx,
                   Grid<GridCoordinate3D> *curTotalEy,
                   Grid<GridCoordinate3D> *curTotalEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffc = curTotalEz->getSize ().getX () / 2;

  GridCoordinateFP3D coordStart (leftNTFF.getX () + 0.5, leftNTFF.getY () + 0.5, z0);
  GridCoordinateFP3D coordEnd (rightNTFF.getX () - 0.5, rightNTFF.getY () - 0.5, z0);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.getX (); coordX <= coordEnd.getX (); ++coordX)
  {
    for (FPValue coordY = coordStart.getY (); coordY <= coordEnd.getY (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0);
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0);
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0);
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0);

      // FieldValue val1 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos1));
      // FieldValue val2 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos2));
      // FieldValue val3 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos3));
      // FieldValue val4 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos4));

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

      FieldValue valEx1 = (curTotalEx->getFieldPointValue (convertCoord (pos1-GridCoordinateFP3D(0,0,0.5)))->getCurValue ()
                           + curTotalEx->getFieldPointValue (convertCoord (pos1+GridCoordinateFP3D(0,0,0.5)))->getCurValue ()) / FPValue(2.0);// - val1;
      FieldValue valEx2 = (curTotalEx->getFieldPointValue (convertCoord (pos2-GridCoordinateFP3D(0,0,0.5)))->getCurValue ()
                           + curTotalEx->getFieldPointValue (convertCoord (pos2+GridCoordinateFP3D(0,0,0.5)))->getCurValue ()) / FPValue(2.0);// - val2;

      FieldValue valEy1 = (curTotalEy->getFieldPointValue (convertCoord (pos3-GridCoordinateFP3D(0,0,0.5)))->getCurValue ()
                           + curTotalEy->getFieldPointValue (convertCoord (pos3+GridCoordinateFP3D(0,0,0.5)))->getCurValue ()) / FPValue(2.0);// - val3;
      FieldValue valEy2 = (curTotalEy->getFieldPointValue (convertCoord (pos4-GridCoordinateFP3D(0,0,0.5)))->getCurValue ()
                           + curTotalEy->getFieldPointValue (convertCoord (pos4+GridCoordinateFP3D(0,0,0.5)))->getCurValue ()) / FPValue(2.0);// - val4;

      FPValue arg = (coordX - diffc) * sin(angleTeta)*cos(anglePhi) + (coordY - diffc) * sin(angleTeta)*sin(anglePhi) + (z0 - diffc) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * (-(valEy1 + valEy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (valEx1 + valEx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * ((valEy1 + valEy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (valEx1 + valEx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN (FPValue angleTeta, FPValue anglePhi,
                 Grid<GridCoordinate3D> *curTotalEz,
                 Grid<GridCoordinate3D> *curTotalHx,
                 Grid<GridCoordinate3D> *curTotalHy,
                 Grid<GridCoordinate3D> *curTotalHz)
{
  return ntffN_x (leftNTFF.getX (), angleTeta, anglePhi, curTotalEz, curTotalHy, curTotalHz)
         + ntffN_x (rightNTFF.getX (), angleTeta, anglePhi, curTotalEz, curTotalHy, curTotalHz)
         + ntffN_y (leftNTFF.getY (), angleTeta, anglePhi, curTotalEz, curTotalHx, curTotalHz)
         + ntffN_y (rightNTFF.getY (), angleTeta, anglePhi, curTotalEz, curTotalHx, curTotalHz)
         + ntffN_z (leftNTFF.getZ (), angleTeta, anglePhi, curTotalEz, curTotalHx, curTotalHy)
         + ntffN_z (rightNTFF.getZ (), angleTeta, anglePhi, curTotalEz, curTotalHx, curTotalHy);
}

Scheme3D::NPair
Scheme3D::ntffL (FPValue angleTeta, FPValue anglePhi,
                 Grid<GridCoordinate3D> *curTotalEx,
                 Grid<GridCoordinate3D> *curTotalEy,
                 Grid<GridCoordinate3D> *curTotalEz)
{
  return ntffL_x (leftNTFF.getX (), angleTeta, anglePhi, curTotalEy, curTotalEz)
         + ntffL_x (rightNTFF.getX (), angleTeta, anglePhi, curTotalEy, curTotalEz)
         + ntffL_y (leftNTFF.getY (), angleTeta, anglePhi, curTotalEx, curTotalEz)
         + ntffL_y (rightNTFF.getY (), angleTeta, anglePhi, curTotalEx, curTotalEz)
         + ntffL_z (leftNTFF.getZ (), angleTeta, anglePhi, curTotalEx, curTotalEy, curTotalEz)
         + ntffL_z (rightNTFF.getZ (), angleTeta, anglePhi, curTotalEx, curTotalEy, curTotalEz);
}

FPValue
Scheme3D::Pointing_scat (FPValue angleTeta, FPValue anglePhi,
                         Grid<GridCoordinate3D> *curTotalEx,
                         Grid<GridCoordinate3D> *curTotalEy,
                         Grid<GridCoordinate3D> *curTotalEz,
                         Grid<GridCoordinate3D> *curTotalHx,
                         Grid<GridCoordinate3D> *curTotalHy,
                         Grid<GridCoordinate3D> *curTotalHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

  NPair N = ntffN (angleTeta, anglePhi, curTotalEz, curTotalHx, curTotalHy, curTotalHz);
  NPair L = ntffL (angleTeta, anglePhi, curTotalEx, curTotalEy, curTotalEz);

  FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);

  FieldValue first = -L.nPhi + n0 * N.nTeta;
  FieldValue second = -L.nTeta - n0 * N.nPhi;

  FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
  FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());

  return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

FPValue
Scheme3D::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
{
  // GridCoordinateFP3D coord (Ez.getSize ().getX () / 2, Ez.getSize ().getY () / 2, Ez.getSize ().getZ () / 2);
  //
  // FieldValue val = approximateIncidentWaveE (coord) * approximateIncidentWaveH (coord) / FPValue(2.0);
  //
  // return val.real ();
  return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0);
}

void
Scheme3D::makeGridScattered (Grid<GridCoordinate3D> *grid, GridType gridType)
{
  for (grid_iter i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    FieldPointValue *val = grid->getFieldPointValue (i);

    GridCoordinate3D pos = grid->calculatePositionFromIndex (i);
    GridCoordinate3D posAbs = grid->getTotalPosition (pos);

    GridCoordinateFP3D realCoord;
    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    GridCoordinateFP3D leftTFSF = convertCoord (yeeLayout->getLeftBorderTFSF ());
    GridCoordinateFP3D rightTFSF = convertCoord (yeeLayout->getRightBorderTFSF ());

    if (realCoord.getX () < leftTFSF.getX ()
        || realCoord.getY () < leftTFSF.getY ()
        || realCoord.getZ () < leftTFSF.getZ ()
        || realCoord.getX () > rightTFSF.getX ()
        || realCoord.getY () > rightTFSF.getY ()
        || realCoord.getZ () > rightTFSF.getZ ())
    {
      continue;
    }

    FieldValue iVal;
    if (gridType == GridType::EX
        || gridType == GridType::EY
        || gridType == GridType::EZ)
    {
      iVal = approximateIncidentWaveE (realCoord);
    }
    else if (gridType == GridType::HX
             || gridType == GridType::HY
             || gridType == GridType::HZ)
    {
      iVal = approximateIncidentWaveH (realCoord);
    }
    else
    {
      UNREACHABLE;
    }

    FieldValue incVal;
    switch (gridType)
    {
      case GridType::EX:
      {
        incVal = yeeLayout->getExFromIncidentE (iVal);
        break;
      }
      case GridType::EY:
      {
        incVal = yeeLayout->getEyFromIncidentE (iVal);
        break;
      }
      case GridType::EZ:
      {
        incVal = yeeLayout->getEzFromIncidentE (iVal);
        break;
      }
      case GridType::HX:
      {
        incVal = yeeLayout->getHxFromIncidentH (iVal);
        break;
      }
      case GridType::HY:
      {
        incVal = yeeLayout->getHyFromIncidentH (iVal);
        break;
      }
      case GridType::HZ:
      {
        incVal = yeeLayout->getHzFromIncidentH (iVal);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    val->setCurValue (val->getCurValue () - incVal);
  }
}

void
Scheme3D::gatherFieldsTotal (bool scattered)
{
  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    if (totalInitialized)
    {
      totalEx = ((ParallelGrid *) Ex)->gatherFullGridPlacement (totalEx);
      totalEy = ((ParallelGrid *) Ey)->gatherFullGridPlacement (totalEy);
      totalEz = ((ParallelGrid *) Ez)->gatherFullGridPlacement (totalEz);

      totalHx = ((ParallelGrid *) Hx)->gatherFullGridPlacement (totalHx);
      totalHy = ((ParallelGrid *) Hy)->gatherFullGridPlacement (totalHy);
      totalHz = ((ParallelGrid *) Hz)->gatherFullGridPlacement (totalHz);
    }
    else
    {
      totalEx = ((ParallelGrid *) Ex)->gatherFullGrid ();
      totalEy = ((ParallelGrid *) Ey)->gatherFullGrid ();
      totalEz = ((ParallelGrid *) Ez)->gatherFullGrid ();

      totalHx = ((ParallelGrid *) Hx)->gatherFullGrid ();
      totalHy = ((ParallelGrid *) Hy)->gatherFullGrid ();
      totalHz = ((ParallelGrid *) Hz)->gatherFullGrid ();

      totalInitialized = true;
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
  else
  {
    if (totalInitialized)
    {
      *totalEx = *Ex;
      *totalEy = *Ey;
      *totalEz = *Ez;

      *totalHx = *Hx;
      *totalHy = *Hy;
      *totalHz = *Hz;
    }
    else
    {
      if (scattered)
      {
        totalEx = new Grid<GridCoordinate3D> (yeeLayout->getExSize (), 0, "Ex");
        totalEy = new Grid<GridCoordinate3D> (yeeLayout->getEySize (), 0, "Ey");
        totalEz = new Grid<GridCoordinate3D> (yeeLayout->getEzSize (), 0, "Ez");

        totalHx = new Grid<GridCoordinate3D> (yeeLayout->getHxSize (), 0, "Hx");
        totalHy = new Grid<GridCoordinate3D> (yeeLayout->getHySize (), 0, "Hy");
        totalHz = new Grid<GridCoordinate3D> (yeeLayout->getHzSize (), 0, "Hz");

        totalInitialized = true;

        *totalEx = *Ex;
        *totalEy = *Ey;
        *totalEz = *Ez;

        *totalHx = *Hx;
        *totalHy = *Hy;
        *totalHz = *Hz;
      }
      else
      {
        totalEx = Ex;
        totalEy = Ey;
        totalEz = Ez;

        totalHx = Hx;
        totalHy = Hy;
        totalHz = Hz;
      }
    }
  }

  if (scattered)
  {
    makeGridScattered (totalEx, GridType::EX);
    makeGridScattered (totalEy, GridType::EY);
    makeGridScattered (totalEz, GridType::EZ);

    makeGridScattered (totalHx, GridType::HX);
    makeGridScattered (totalHy, GridType::HY);
    makeGridScattered (totalHz, GridType::HZ);
  }
}

void
Scheme3D::saveGrids (time_step t)
{
  int processId = 0;

  GridCoordinate3D startEx (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinExCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinExCoordFP ().getY ()) + 1,
                            Ex->getSize ().getZ () / 2);
  GridCoordinate3D endEx (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinExCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinExCoordFP ().getY ()),
                          Ex->getSize ().getZ () / 2 + 1);

  GridCoordinate3D startEy (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinEyCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinEyCoordFP ().getY ()) + 1,
                            Ey->getSize ().getZ () / 2);
  GridCoordinate3D endEy (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinEyCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinEyCoordFP ().getY ()),
                          Ey->getSize ().getZ () / 2 + 1);

  GridCoordinate3D startEz (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinEzCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinEzCoordFP ().getY ()) + 1,
                            Ez->getSize ().getZ () / 2);
  GridCoordinate3D endEz (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinEzCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinEzCoordFP ().getY ()),
                          Ez->getSize ().getZ () / 2 + 1);

  GridCoordinate3D startHx (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinHxCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinHxCoordFP ().getY ()) + 1,
                            Hx->getSize ().getZ () / 2);
  GridCoordinate3D endHx (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinHxCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinHxCoordFP ().getY ()),
                          Hx->getSize ().getZ () / 2 + 1);

  GridCoordinate3D startHy (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinHyCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinHyCoordFP ().getY ()) + 1,
                            Hy->getSize ().getZ () / 2);
  GridCoordinate3D endHy (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinHyCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinHyCoordFP ().getY ()),
                          Hy->getSize ().getZ () / 2 + 1);

  GridCoordinate3D startHz (grid_coord (yeeLayout->getLeftBorderPML ().getX () - yeeLayout->getMinHzCoordFP ().getX ()) + 1,
                            grid_coord (yeeLayout->getLeftBorderPML ().getY () - yeeLayout->getMinHzCoordFP ().getY ()) + 1,
                            Hz->getSize ().getZ () / 2);
  GridCoordinate3D endHz (grid_coord (yeeLayout->getRightBorderPML ().getX () - yeeLayout->getMinHzCoordFP ().getX ()),
                          grid_coord (yeeLayout->getRightBorderPML ().getY () - yeeLayout->getMinHzCoordFP ().getY ()),
                          Hz->getSize ().getZ () / 2);

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Ex");
    dumper[type]->dumpGrid (totalEx, startEx, endEx);

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Ey");
    dumper[type]->dumpGrid (totalEy, startEy, endEy);

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Ez");
    dumper[type]->dumpGrid (totalEz, startEz, endEz);

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Hx");
    dumper[type]->dumpGrid (totalHx, startHx, endHx);

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Hy");
    dumper[type]->dumpGrid (totalHy, startHy, endHy);

    dumper[type]->init (t, CURRENT, processId, "3D-in-time-Hz");
    dumper[type]->dumpGrid (totalHz, startHz, endHz);
  }
}

void
Scheme3D::saveNTFF (bool isReverse, time_step t)
{
#ifdef PARALLEL_GRID
  int processId = ParallelGrid::getParallelCore ()->getProcessId ();

  if (processId == 0)
#endif
  {
    std::ofstream outfile;
    outfile.open (solverSettings.getFileNameNTFF ().c_str ());
    FPValue start;
    FPValue end;
    FPValue step;

    if (isReverse)
    {
      outfile << "Reverse diagram" << std::endl << std::endl;
      start = yeeLayout->getIncidentWaveAngle2 ();
      end = yeeLayout->getIncidentWaveAngle2 ();
      step = 1.0;
    }
    else
    {
      outfile << "Forward diagram" << std::endl << std::endl;
      start = 0.0;
      end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
      step = PhysicsConst::Pi * solverSettings.getAngleStepNTFF () / 180;
    }

    for (FPValue angle = start; angle <= end; angle += step)
    {
      FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
                                   angle,
                                   totalEx,
                                   totalEy,
                                   totalEz,
                                   totalHx,
                                   totalHy,
                                   totalHz) / Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);

      outfile << "timestep = "
              << t
              << ", incident wave angle=("
              << yeeLayout->getIncidentWaveAngle1 () << ","
              << yeeLayout->getIncidentWaveAngle2 () << ","
              << yeeLayout->getIncidentWaveAngle3 () << ","
              << "), angle NTFF = "
              << angle
              << ", NTFF value = "
              << val
              << std::endl;
    }

    outfile.close ();
  }
}

void Scheme3D::additionalUpdateOfGrids (time_step t, time_step &diffT)
{
  if (solverSettings.getDoUseParallelGrid () && solverSettings.getDoUseDynamicGrid ())
  {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
    //if (false && t % solverSettings.getRebalanceStep () == 0)
    if (t % diffT == 0 && t > 0)
    {
      if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
      {
        DPRINTF (LOG_LEVEL_STAGES, "Try rebalance on step %u, steps elapsed after previous %u\n", t, diffT);
      }
      ParallelGrid::getParallelCore ()->ShareClocks ();

      ParallelYeeGridLayout *parallelYeeLayout = (ParallelYeeGridLayout *) yeeLayout;

      if (parallelYeeLayout->Rebalance (diffT))
      {
        DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalancing for process %d!\n", ParallelGrid::getParallelCore ()->getProcessId ());

        ((ParallelGrid *) Eps)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        ((ParallelGrid *) Mu)->Resize (parallelYeeLayout->getMuSizeForCurNode ());

        ((ParallelGrid *) Ex)->Resize (parallelYeeLayout->getExSizeForCurNode ());
        ((ParallelGrid *) Ey)->Resize (parallelYeeLayout->getEySizeForCurNode ());
        ((ParallelGrid *) Ez)->Resize (parallelYeeLayout->getEzSizeForCurNode ());

        ((ParallelGrid *) Hx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
        ((ParallelGrid *) Hy)->Resize (parallelYeeLayout->getHySizeForCurNode ());
        ((ParallelGrid *) Hz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());

        if (solverSettings.getDoUsePML ())
        {
          ((ParallelGrid *) Dx)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          ((ParallelGrid *) Dy)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          ((ParallelGrid *) Dz)->Resize (parallelYeeLayout->getEzSizeForCurNode ());

          ((ParallelGrid *) Bx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          ((ParallelGrid *) By)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          ((ParallelGrid *) Bz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());

          if (solverSettings.getDoUseMetamaterials ())
          {
            ((ParallelGrid *) D1x)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            ((ParallelGrid *) D1y)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            ((ParallelGrid *) D1z)->Resize (parallelYeeLayout->getEzSizeForCurNode ());

            ((ParallelGrid *) B1x)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            ((ParallelGrid *) B1y)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            ((ParallelGrid *) B1z)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
          }

          ((ParallelGrid *) SigmaX)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) SigmaY)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) SigmaZ)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        }

        if (solverSettings.getDoUseAmplitudeMode ())
        {
          ((ParallelGrid *) ExAmplitude)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          ((ParallelGrid *) EyAmplitude)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          ((ParallelGrid *) EzAmplitude)->Resize (parallelYeeLayout->getEzSizeForCurNode ());

          ((ParallelGrid *) HxAmplitude)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          ((ParallelGrid *) HyAmplitude)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          ((ParallelGrid *) HzAmplitude)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
        }

        if (solverSettings.getDoUseMetamaterials ())
        {
          ((ParallelGrid *) OmegaPE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) GammaE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) OmegaPM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) GammaM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        }

        ParallelGrid::getParallelCore ()->ClearClocks ();

        //diffT += 1;
        //diffT *= 2;
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

#endif /* GRID_3D */

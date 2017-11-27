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
  , DInc (NULLPTR)
  , HInc (NULLPTR)
  , BInc (NULLPTR)
  , SigmaXInc (NULLPTR)
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
  , ExBorder (NULLPTR)
  , ExInitial (NULLPTR)
  , EyBorder (NULLPTR)
  , EyInitial (NULLPTR)
  , EzBorder (NULLPTR)
  , EzInitial (NULLPTR)
  , HxBorder (NULLPTR)
  , HxInitial (NULLPTR)
  , HyBorder (NULLPTR)
  , HyInitial (NULLPTR)
  , HzBorder (NULLPTR)
  , HzInitial (NULLPTR)
  , Jx (NULLPTR)
  , Jy (NULLPTR)
  , Jz (NULLPTR)
  , Mx (NULLPTR)
  , My (NULLPTR)
  , Mz (NULLPTR)
  , ExExact (NULLPTR)
  , EyExact (NULLPTR)
  , EzExact (NULLPTR)
  , HxExact (NULLPTR)
  , HyExact (NULLPTR)
  , HzExact (NULLPTR)
  , normG (0)
{
  if (solverSettings.getDoUseNTFF ())
  {
    leftNTFF = GridCoordinate3D (solverSettings.getNTFFSizeX (), solverSettings.getNTFFSizeY (), solverSettings.getNTFFSizeZ ());
    rightNTFF = layout->getEzSize () - leftNTFF + GridCoordinate3D (1,1,1);
  }

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
    if (solverSettings.getDoUseTFSFPML ())
    {
      EInc = new Grid<GridCoordinate1D> (GridCoordinate1D (1000), 0, "EInc");
      HInc = new Grid<GridCoordinate1D> (GridCoordinate1D (1000), 0, "HInc");

      DInc = new Grid<GridCoordinate1D> (GridCoordinate1D (1000), 0, "DInc");
      BInc = new Grid<GridCoordinate1D> (GridCoordinate1D (1000), 0, "BInc");

      SigmaXInc = new Grid<GridCoordinate1D> (GridCoordinate1D (1000), 0, "SigmaXInc");
    }
    else
    {
      EInc = new Grid<GridCoordinate1D> (GridCoordinate1D (100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0, "EInc");
      HInc = new Grid<GridCoordinate1D> (GridCoordinate1D (100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0, "HInc");
    }
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

    dumper[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate3D> ();
    ((BMPDumper<GridCoordinate3D> *) dumper[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);

    dumper1D[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate1D> ();
    ((BMPDumper<GridCoordinate1D> *) dumper1D[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  else
  {
    dumper[FILE_TYPE_BMP] = NULLPTR;
    dumper1D[FILE_TYPE_BMP] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsDAT ())
  {
    dumper[FILE_TYPE_DAT] = new DATDumper<GridCoordinate3D> ();
    dumper1D[FILE_TYPE_DAT] = new DATDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_DAT] = NULLPTR;
    dumper1D[FILE_TYPE_DAT] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsTXT ())
  {
    dumper[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate3D> ();
    dumper1D[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_TXT] = NULLPTR;
    dumper1D[FILE_TYPE_TXT] = NULLPTR;
  }

  if (!solverSettings.getEpsFileName ().empty ()
      || !solverSettings.getMuFileName ().empty ()
      || !solverSettings.getOmegaPEFileName ().empty ()
      || !solverSettings.getOmegaPMFileName ().empty ()
      || !solverSettings.getGammaEFileName ().empty ()
      || !solverSettings.getGammaMFileName ().empty ())
  {
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

      ((BMPLoader<GridCoordinate3D> *) loader[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
    }
    {
      loader[FILE_TYPE_DAT] = new DATLoader<GridCoordinate3D> ();
    }
    {
      loader[FILE_TYPE_TXT] = new TXTLoader<GridCoordinate3D> ();
    }
  }
  else
  {
    loader[FILE_TYPE_BMP] = NULLPTR;
    loader[FILE_TYPE_DAT] = NULLPTR;
    loader[FILE_TYPE_TXT] = NULLPTR;
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

    delete DInc;
    delete BInc;

    delete SigmaXInc;
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

  delete dumper1D[FILE_TYPE_BMP];
  delete dumper1D[FILE_TYPE_DAT];
  delete dumper1D[FILE_TYPE_TXT];
}

void
Scheme3D::performPlaneWaveDSteps (time_step t)
{
  grid_coord size = DInc->getSize ().getX ();

  ASSERT (size > 0);

  FPValue modifier = 1 / (relPhaseVelocity * gridStep);

  for (grid_coord i = 1; i < size; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valD = DInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i - 1);
    GridCoordinate1D posRight (i);

    FieldPointValue *valH1 = HInc->getFieldPointValue (posLeft);
    FieldPointValue *valH2 = HInc->getFieldPointValue (posRight);

    FieldPointValue *valSigma = SigmaXInc->getFieldPointValue (pos);

    FPValue Ca = (2 * PhysicsConst::Eps0 - valSigma->getCurValue () * gridTimeStep)
                 / (2 * PhysicsConst::Eps0 + valSigma->getCurValue () * gridTimeStep);
    FPValue Cb =  2 * PhysicsConst::Eps0 * gridTimeStep / (2 * PhysicsConst::Eps0 + valSigma->getCurValue () * gridTimeStep);

    FieldValue val = Ca * valD->getPrevValue () + Cb * modifier * (valH2->getPrevValue () - valH1->getPrevValue ());

    if (t == 0)
    {
    //   FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;
    //
    //   // exp (- (x-x0)^2)
    //
    // #ifdef COMPLEX_FIELD_VALUES
    //   valE->setCurValue (FieldValue (sin (arg), cos (arg)));
    // #else /* COMPLEX_FIELD_VALUES */
    //   //valE->setCurValue (sin (arg));
      // valD->setCurValue (PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * (exp (- SQR((FPValue) (pos.getX() - 500 + (0.5)*1.0 / 2.0)) / SQR(15.0)/64)
      //                                          + exp (- SQR((FPValue) (pos.getX() - 500 - (0.5)*1.0 / 2.0)) / SQR(15.0)/64)) / 2);
      // val = PhysicsConst::Eps0 * sin(gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
      //                                + 2 * PhysicsConst::Pi / sourceWaveLength * (i) * gridStep);
      val = PhysicsConst::Eps0 * (exp (-SQR((i-500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  + exp (-SQR((i-500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
    // #endif /* !COMPLEX_FIELD_VALUES */
    }

    valD->setCurValue (val);

    //if (i >= solverSettings.getTFSFPMLSizeXLeft () && i <= size - solverSettings.getTFSFPMLSizeXRight ())
    {
      FPValue exact = PhysicsConst::Eps0 * (exp (-SQR((i-500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  + exp (-SQR((i-500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
      // FPValue exact = PhysicsConst::Eps0 * sin(gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
      //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i) * gridStep);
      // exact solution diff
      // FPValue exact = (PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * (exp (- SQR((FPValue) (pos.getX() - 500 + (t+0.5)*1.0 / 2.0)) / SQR(15.0)/64)
      //                  + exp (- SQR((FPValue) (pos.getX() - 500 - (t+0.5)*1.0 / 2.0)) / SQR(15.0)/64)) / 2);
      // if (fabs(exact - valE->getCurValue ()) > norm)
      // {
      //   norm = fabs(exact - valE->getCurValue ());
      // }
      FPValue norm = SQR (exact - valD->getCurValue ());
    }
  }

  //ALWAYS_ASSERT (DInc->getFieldPointValue (GridCoordinate1D (size - 1))->getCurValue () == getFieldValueRealOnly (0.0));

  FPValue val1 = PhysicsConst::Eps0 * (exp (-SQR((0 - 500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              + exp (-SQR((0 - 500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  FPValue val2 = PhysicsConst::Eps0 * (exp (-SQR((size - 1 - 500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                      + exp (-SQR((size - 1 - 500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  // FPValue arg1 = gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (0) * gridStep;
  // FPValue arg2 = gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (size - 1) * gridStep;

  DInc->getFieldPointValue (GridCoordinate1D(0))->setCurValue (val1);
  DInc->getFieldPointValue (GridCoordinate1D(size - 1))->setCurValue (val2);

  DInc->nextTimeStep ();
}

void
Scheme3D::performPlaneWaveESteps (time_step t)
{
  grid_coord size = EInc->getSize ().getX ();

  ASSERT (size > 0);

  FPValue norm = 0.0;

  for (grid_coord i = 1; i < size; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valE = EInc->getFieldPointValue (pos);
    FieldPointValue *valD = DInc->getFieldPointValue (pos);

    FPValue eps = 1.0;
    // if (i >= size / 2 - 100 && i < size / 2 + 100)
    // {
    //   eps = 4.0;
    // }

    FieldValue val = valE->getPrevValue () + 1 / (eps * PhysicsConst::Eps0) * (valD->getPrevValue () - valD->getPrevPrevValue ());

    if (t == 0)
    {
    //   FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;
    //
    //   // exp (- (x-x0)^2)
    //
    // #ifdef COMPLEX_FIELD_VALUES
    //   valE->setCurValue (FieldValue (sin (arg), cos (arg)));
    // #else /* COMPLEX_FIELD_VALUES */
    //   //valE->setCurValue (sin (arg));
      //valE->setCurValue (PhysicsConst::SpeedOfLight * (exp (- SQR((FPValue) (pos.getX() - 500 + (0.5)*1.0 / 2.0)) / SQR(15.0)/64)
        //                 + exp (- SQR((FPValue) (pos.getX() - 500 - (0.5)*1.0 / 2.0)) / SQR(15.0)/64)) / 2);
        // val = (sin(gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
        //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i) * gridStep));
        val = (exp (-SQR((i-500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
               + exp (-SQR((i-500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
    // #endif /* !COMPLEX_FIELD_VALUES */
    }

    valE->setCurValue (val);

    //if (i >= solverSettings.getTFSFPMLSizeXLeft () && i <= size - solverSettings.getTFSFPMLSizeXRight ())
    {
      FPValue exact = (exp (-SQR((i-500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  + exp (-SQR((i-500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
      // FPValue exact = sin(gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
      //                     + 2 * PhysicsConst::Pi / sourceWaveLength * (i) * gridStep);
      //
      // exact solution diff
      // FPValue exact = PhysicsConst::SpeedOfLight * (exp (- SQR((FPValue) (pos.getX() - 500 + (t+0.5)*1.0 / 2.0)) / SQR(15.0)/64)
      //                  + exp (- SQR((FPValue) (pos.getX() - 500 - (t+0.5)*1.0 / 2.0)) / SQR(15.0)/64)) / 2;
      // if (fabs(exact - valE->getCurValue ()) > norm)
      // {
      //   norm = fabs(exact - valE->getCurValue ());
      // }
      norm += SQR (exact - valE->getCurValue ());
    }
  }

  norm = sqrt(norm / size);

  if (norm > normG)
  {
    normG = norm;
  }

  printf ("NORM: t=%u, %.30f, %.30f\n", t, norm, normG);


  FPValue val1 = (exp (-SQR((0 - 500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              + exp (-SQR((0 - 500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  FPValue val2 = (exp (-SQR((size - 1 - 500)*gridStep + (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              + exp (-SQR((size - 1 - 500)*gridStep - (t+0.5)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  // FPValue arg1 = gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (0) * gridStep;
  // FPValue arg2 = gridTimeStep * (t + 0.5) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (size - 1) * gridStep;

  // exp (- (x-x0)^2)

// #ifdef COMPLEX_FIELD_VALUES
//   // EInc->getFieldPointValue (GridCoordinate1D(0))->setCurValue (FieldValue (sin (arg), cos (arg)));
//   // EInc->getFieldPointValue (GridCoordinate1D(size - 1))->setCurValue (FieldValue (sin (arg), cos (arg)));
// #else /* COMPLEX_FIELD_VALUES */
  EInc->getFieldPointValue (GridCoordinate1D(0))->setCurValue (val1);
  EInc->getFieldPointValue (GridCoordinate1D(size - 1))->setCurValue (val2);
// #endif /* !COMPLEX_FIELD_VALUES */

  //ALWAYS_ASSERT (EInc->getFieldPointValue (GridCoordinate1D (size - 1))->getCurValue () == getFieldValueRealOnly (0.0));

  EInc->nextTimeStep ();
}

void
Scheme3D::performPlaneWaveBSteps (time_step t)
{
  grid_coord size = BInc->getSize ().getX ();

  ASSERT (size > 0);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * gridStep);

  for (grid_coord i = 0; i < size - 1; ++i)
  {
    GridCoordinate1D pos (i);

    FieldPointValue *valB = BInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i);
    GridCoordinate1D posRight (i+1);

    FieldPointValue *valE1 = EInc->getFieldPointValue (posLeft);
    FieldPointValue *valE2 = EInc->getFieldPointValue (posRight);

    FieldValue val = valB->getPrevValue () + modifier * (valE2->getPrevValue () - valE1->getPrevValue ());

    if (t == 0)
    {
    //   FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;
    //
    //   // exp (- (x-x0)^2)
    //
    // #ifdef COMPLEX_FIELD_VALUES
    //   valE->setCurValue (FieldValue (sin (arg), cos (arg)));
    // #else /* COMPLEX_FIELD_VALUES */
    //   //valE->setCurValue (sin (arg));
      // valB->setCurValue (PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * PhysicsConst::Mu0 * (exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 + (1)*1.0 / 2.0)) / SQR(15.0)/64)
      //                     - exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 - (1)*1.0 / 2.0)) / SQR(15.0)/64)) / 2);
      // val = sqrt(PhysicsConst::Eps0*PhysicsConst::Mu0) * sin(gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
      //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i + 0.5) * gridStep);
      val = sqrt(PhysicsConst::Eps0 * PhysicsConst::Mu0) * (exp (-SQR((i+0.5-500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  - exp (-SQR((i+0.5-500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
    // #endif /* !COMPLEX_FIELD_VALUES */
    }

    valB->setCurValue (val);

    //if (i >= solverSettings.getTFSFPMLSizeXLeft () && i <= size - solverSettings.getTFSFPMLSizeXRight ())
    {
      FPValue exact = sqrt(PhysicsConst::Eps0 * PhysicsConst::Mu0) * (exp (-SQR((i+0.5-500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  - exp (-SQR((i+0.5-500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
      // FPValue exact = sqrt(PhysicsConst::Eps0*PhysicsConst::Mu0) * sin(gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
      //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i + 0.5) * gridStep);
      // exact solution diff
      // FPValue exact = PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * PhysicsConst::Mu0 * (exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 + (t+1)*1.0 / 2.0)) / SQR(15.0)/64)
      //                  - exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 - (t+1)*1.0 / 2.0)) / SQR(15.0)/64)) / 2;
      // if (fabs(exact - valE->getCurValue ()) > norm)
      // {
      //   norm = fabs(exact - valE->getCurValue ());
      // }
      FPValue norm = SQR (exact - valB->getCurValue ());
    }
  }

  FPValue val1 = sqrt(PhysicsConst::Eps0 * PhysicsConst::Mu0) * (exp (-SQR((0 + 0.5 - 500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              - exp (-SQR((0 + 0.5 - 500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  FPValue val2 = sqrt(PhysicsConst::Eps0 * PhysicsConst::Mu0) * (exp (-SQR((size - 1 + 0.5 - 500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              - exp (-SQR((size - 1 + 0.5 - 500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  // FPValue arg1 = gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (0 + 0.5) * gridStep;
  // FPValue arg2 = gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (size + 0.5 - 1) * gridStep;
  BInc->getFieldPointValue (GridCoordinate1D(0))->setCurValue (val1);
  BInc->getFieldPointValue (GridCoordinate1D(size - 1))->setCurValue (val2);

  BInc->nextTimeStep ();
}

void
Scheme3D::performPlaneWaveHSteps (time_step t)
{
  grid_coord size = HInc->getSize ().getX ();

  ASSERT (size > 0);

  for (grid_coord i = 0; i < size - 1; ++i)
  {
    GridCoordinate1D pos (i);

    GridCoordinate1D posLeft (i);
    GridCoordinate1D posRight (i+1);

    FieldPointValue *valH = HInc->getFieldPointValue (pos);
    FieldPointValue *valB = BInc->getFieldPointValue (pos);

    FieldPointValue *valSigma1 = SigmaXInc->getFieldPointValue (posLeft);
    FieldPointValue *valSigma2 = SigmaXInc->getFieldPointValue (posRight);
    FieldValue sigma = (valSigma1->getCurValue () + valSigma2->getCurValue ()) / 2;

    // FPValue eps = 1.0;
    // if (i == size / 2 - 1)
    // {
    //   eps = 2.5;
    // }
    // else if (i > size / 2 - 1)
    // {
    //   eps = 4.0;
    // }
    FPValue mu = 1.0;

    FPValue Da = (2 * PhysicsConst::Eps0 - sigma * gridTimeStep)
                 / (2 * PhysicsConst::Eps0 + sigma * gridTimeStep);
    FPValue Db =  2 * PhysicsConst::Eps0 / (2 * PhysicsConst::Eps0 + sigma * gridTimeStep);

    FieldValue val = Da * valH->getPrevValue () + Db / (mu * PhysicsConst::Mu0) * (valB->getPrevValue () - valB->getPrevPrevValue ());

    if (t == 0)
    {
    //   FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;
    //
    //   // exp (- (x-x0)^2)
    //
    // #ifdef COMPLEX_FIELD_VALUES
    //   valE->setCurValue (FieldValue (sin (arg), cos (arg)));
    // #else /* COMPLEX_FIELD_VALUES */
    //   //valE->setCurValue (sin (arg));
      // valH->setCurValue (PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * (exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 + (1)*1.0 / 2.0)) / SQR(15.0)/64)
      //                     - exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 - (1)*1.0 / 2.0)) / SQR(15.0)/64)) / 2);
      // val = (sqrt(PhysicsConst::Eps0/PhysicsConst::Mu0) * sin(gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
      //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i + 0.5) * gridStep));
      val = sqrt(PhysicsConst::Eps0 / PhysicsConst::Mu0) * (exp (-SQR((i+0.5-500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  - exp (-SQR((i+0.5-500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
    // #endif /* !COMPLEX_FIELD_VALUES */
    }

    valH->setCurValue (val);

    // if (i >= solverSettings.getTFSFPMLSizeXLeft () && i <= size - solverSettings.getTFSFPMLSizeXRight ())
    {
      FPValue exact = sqrt(PhysicsConst::Eps0 / PhysicsConst::Mu0) * (exp (-SQR((i+0.5-500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                                  - exp (-SQR((i+0.5-500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
      // FPValue exact = sqrt(PhysicsConst::Eps0/PhysicsConst::Mu0) * sin(gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
      //                        + 2 * PhysicsConst::Pi / sourceWaveLength * (i + 0.5) * gridStep);
      // exact solution diff
      // FPValue exact = PhysicsConst::SpeedOfLight * PhysicsConst::Eps0 * (exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 + (t+1)*1.0 / 2.0)) / SQR(15.0)/64)
      //                  - exp (- SQR((FPValue) (pos.getX() + 0.5 - 500 - (t+1)*1.0 / 2.0)) / SQR(15.0)/64)) / 2;
      // if (fabs(exact - valE->getCurValue ()) > norm)
      // {
      //   norm = fabs(exact - valE->getCurValue ());
      // }
      FPValue norm = SQR (exact - valH->getCurValue ());
    }
  }

  FPValue val1 = sqrt(PhysicsConst::Eps0 / PhysicsConst::Mu0) * (exp (-SQR((0+0.5-500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              - exp (-SQR((0+0.5-500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  FPValue val2 = sqrt(PhysicsConst::Eps0 / PhysicsConst::Mu0) * (exp (-SQR((size - 1 + 0.5 - 500)*gridStep + (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))
                              - exp (-SQR((size - 1 + 0.5 - 500)*gridStep - (t+1)*gridTimeStep*PhysicsConst::SpeedOfLight)/SQR(15.0*16*gridStep))) / 2.0;
  // FPValue arg1 = gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (0 + 0.5) * gridStep;
  // FPValue arg2 = gridTimeStep * (t + 1) * 2 * PhysicsConst::Pi * sourceFrequency
  //                + 2 * PhysicsConst::Pi / sourceWaveLength * (size + 0.5 - 1) * gridStep;
  HInc->getFieldPointValue (GridCoordinate1D(0))->setCurValue (val1);
  HInc->getFieldPointValue (GridCoordinate1D(size - 1))->setCurValue (val2);


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
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EX), true, true> (t, ExStart, ExEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EX), true, false> (t, ExStart, ExEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EX), false, true> (t, ExStart, ExEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EX), false, false> (t, ExStart, ExEnd);
    }
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
  FPValue coordD1 = (FPValue) ((grid_coord) d);
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
Scheme3D::performEySteps (time_step t, GridCoordinate3D EyStart, GridCoordinate3D EyEnd)
{
  /*
   * FIXME: check performed on each iteration
   */
  if (solverSettings.getDoUsePML ())
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EY), true, true> (t, EyStart, EyEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EY), true, false> (t, EyStart, EyEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EY), false, true> (t, EyStart, EyEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EY), false, false> (t, EyStart, EyEnd);
    }
  }

  if (solverSettings.getDoUsePointSourceEy ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::EY)> (t);
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
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EZ), true, true> (t, EzStart, EzEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EZ), true, false> (t, EzStart, EzEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EZ), false, true> (t, EzStart, EzEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::EZ), false, false> (t, EzStart, EzEnd);
    }
  }

  if (solverSettings.getDoUsePointSourceEz ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::EZ)> (t);
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
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HX), true, true> (t, HxStart, HxEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HX), true, false> (t, HxStart, HxEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HX), false, true> (t, HxStart, HxEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HX), false, false> (t, HxStart, HxEnd);
    }
  }

  if (solverSettings.getDoUsePointSourceHx ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HX)> (t);
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
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HY), true, true> (t, HyStart, HyEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HY), true, false> (t, HyStart, HyEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HY), false, true> (t, HyStart, HyEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HY), false, false> (t, HyStart, HyEnd);
    }
  }

  if (solverSettings.getDoUsePointSourceHy ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HY)> (t);
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
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HZ), true, true> (t, HzStart, HzEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HZ), true, false> (t, HzStart, HzEnd);
    }
  }
  else
  {
    if (solverSettings.getDoUseMetamaterials ())
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HZ), false, true> (t, HzStart, HzEnd);
    }
    else
    {
      calculateFieldStep<static_cast<uint8_t> (GridType::HZ), false, false> (t, HzStart, HzEnd);
    }
  }

  if (solverSettings.getDoUsePointSourceHz ())
  {
    performPointSourceCalc<static_cast<uint8_t> (GridType::HZ)> (t);
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
      performPlaneWaveDSteps (t);
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
      performPlaneWaveBSteps (t);
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
        && t > 0 && t % solverSettings.getIntermediateNTFFStep () == 0)
    {
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

    DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " FP_MOD ". \n", t, maxAccuracy);
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
Scheme3D::initScheme (FPValue dx, FPValue sourceWaveLen)
{
  sourceWaveLength = sourceWaveLen;
  sourceFrequency = PhysicsConst::SpeedOfLight / sourceWaveLength;

  gridStep = dx;
  courantNum = solverSettings.getCourantNum ();
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  FPValue phaseVelocity0 = Approximation::phaseVelocityIncidentWave3D (gridStep, sourceWaveLength, courantNum, N_lambda, PhysicsConst::Pi / 2, 0);
  FPValue phaseVelocity = Approximation::phaseVelocityIncidentWave3D (gridStep, sourceWaveLength, courantNum, N_lambda, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ());

  relPhaseVelocity = phaseVelocity0 / phaseVelocity;
}

void
Scheme3D::initCallBacks ()
{
#ifndef COMPLEX_FIELD_VALUES
  if (solverSettings.getDoUsePolinom1BorderCondition ())
  {
    EzBorder = CallBack::polinom1_ez;
    HyBorder = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoUsePolinom2BorderCondition ())
  {
    ExBorder = CallBack::polinom2_ex;
    EyBorder = CallBack::polinom2_ey;
    EzBorder = CallBack::polinom2_ez;

    HxBorder = CallBack::polinom2_hx;
    HyBorder = CallBack::polinom2_hy;
    HzBorder = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoUsePolinom3BorderCondition ())
  {
    EzBorder = CallBack::polinom3_ez;
    HyBorder = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoUseSin1BorderCondition ())
  {
    EzBorder = CallBack::sin1_ez;
    HyBorder = CallBack::sin1_hy;
  }

  if (solverSettings.getDoUsePolinom1StartValues ())
  {
    EzInitial = CallBack::polinom1_ez;
    HyInitial = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoUsePolinom2StartValues ())
  {
    ExInitial = CallBack::polinom2_ex;
    EyInitial = CallBack::polinom2_ey;
    EzInitial = CallBack::polinom2_ez;

    HxInitial = CallBack::polinom2_hx;
    HyInitial = CallBack::polinom2_hy;
    HzInitial = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoUsePolinom3StartValues ())
  {
    EzInitial = CallBack::polinom3_ez;
    HyInitial = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoUseSin1StartValues ())
  {
    EzInitial = CallBack::sin1_ez;
    HyInitial = CallBack::sin1_hy;
  }

  if (solverSettings.getDoUsePolinom1RightSide ())
  {
    Jz = CallBack::polinom1_jz;
    My = CallBack::polinom1_my;
  }
  else if (solverSettings.getDoUsePolinom2RightSide ())
  {
    Jx = CallBack::polinom2_jx;
    Jy = CallBack::polinom2_jy;
    Jz = CallBack::polinom2_jz;

    Mx = CallBack::polinom2_mx;
    My = CallBack::polinom2_my;
    Mz = CallBack::polinom2_mz;
  }
  else if (solverSettings.getDoUsePolinom3RightSide ())
  {
    Jz = CallBack::polinom3_jz;
    My = CallBack::polinom3_my;
  }

  if (solverSettings.getDoCalculatePolinom1DiffNorm ())
  {
    EzExact = CallBack::polinom1_ez;
    HyExact = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoCalculatePolinom2DiffNorm ())
  {
    ExExact = CallBack::polinom2_ex;
    EyExact = CallBack::polinom2_ey;
    EzExact = CallBack::polinom2_ez;

    HxExact = CallBack::polinom2_hx;
    HyExact = CallBack::polinom2_hy;
    HzExact = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoCalculatePolinom3DiffNorm ())
  {
    EzExact = CallBack::polinom3_ez;
    HyExact = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoCalculateSin1DiffNorm ())
  {
    EzExact = CallBack::sin1_ez;
    HyExact = CallBack::sin1_hy;
  }
#endif
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
          val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                      center * modifier + GridCoordinateFP3D (0.5, 0.5, 0.5),
                                                                      solverSettings.getEpsSphereRadius () * modifier,
                                                                      epsVal));
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

    if (solverSettings.getOmegaPESphere () != 0)
    {
      for (int i = 0; i < OmegaPE->getSize ().getX (); ++i)
      {
        for (int j = 0; j < OmegaPE->getSize ().getY (); ++j)
        {
          for (int k = 0; k < OmegaPE->getSize ().getZ (); ++k)
          {
            GridCoordinate3D pos (i, j, k);
            GridCoordinateFP3D posAbs = yeeLayout->getEpsCoordFP (OmegaPE->getTotalPosition (pos));
            FieldPointValue *val = OmegaPE->getFieldPointValue (pos);

            FieldValue omegapeVal = getFieldValueRealOnly (solverSettings.getOmegaPESphere () * 2 * PhysicsConst::Pi * sourceFrequency);

            FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);
            GridCoordinateFP3D center (solverSettings.getOmegaPESphereCenterX (),
                                       solverSettings.getOmegaPESphereCenterY (),
                                       solverSettings.getOmegaPESphereCenterZ ());
            val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                        center * modifier + GridCoordinateFP3D (0.5, 0.5, 0.5),
                                                                        solverSettings.getOmegaPESphereRadius () * modifier,
                                                                        omegapeVal));
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
                                getStartCoord (GridType::EPS, totalEps->getSize ()),
                                getEndCoord (GridType::EPS, totalEps->getSize ()));

        dumper[type]->init (0, CURRENT, processId, "Mu");
        dumper[type]->dumpGrid (totalMu,
                                getStartCoord (GridType::MU, totalMu->getSize ()),
                                getEndCoord (GridType::MU, totalMu->getSize ()));

        if (solverSettings.getDoUseMetamaterials ())
        {
          dumper[type]->init (0, CURRENT, processId, "OmegaPE");
          dumper[type]->dumpGrid (totalOmegaPE,
                                  getStartCoord (GridType::OMEGAPE, totalOmegaPE->getSize ()),
                                  getEndCoord (GridType::OMEGAPE, totalOmegaPE->getSize ()));

          dumper[type]->init (0, CURRENT, processId, "OmegaPM");
          dumper[type]->dumpGrid (totalOmegaPM,
                                  getStartCoord (GridType::OMEGAPM, totalOmegaPM->getSize ()),
                                  getEndCoord (GridType::OMEGAPM, totalOmegaPM->getSize ()));

          dumper[type]->init (0, CURRENT, processId, "GammaE");
          dumper[type]->dumpGrid (totalGammaE,
                                  getStartCoord (GridType::GAMMAE, totalGammaE->getSize ()),
                                  getEndCoord (GridType::GAMMAE, totalGammaE->getSize ()));

          dumper[type]->init (0, CURRENT, processId, "GammaM");
          dumper[type]->dumpGrid (totalGammaM,
                                  getStartCoord (GridType::GAMMAM, totalGammaM->getSize ()),
                                  getEndCoord (GridType::GAMMAM, totalGammaM->getSize ()));
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

  if (ExInitial != NULLPTR)
  {
    for (int i = 0; i < Ex->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ex->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ex->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Ex->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (posAbs);

          Ex->getFieldPointValue (pos)->setCurValue (ExInitial (realCoord * gridStep, 0.5 * gridTimeStep));
        }
      }
    }
  }

  if (EyInitial != NULLPTR)
  {
    for (int i = 0; i < Ey->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ey->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ey->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Ey->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (posAbs);

          Ey->getFieldPointValue (pos)->setCurValue (EyInitial (realCoord * gridStep, 0.5 * gridTimeStep));
        }
      }
    }
  }

  if (EzInitial != NULLPTR)
  {
    for (int i = 0; i < Ez->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ez->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ez->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Ez->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (posAbs);

          Ez->getFieldPointValue (pos)->setCurValue (EzInitial (realCoord * gridStep, 0.5 * gridTimeStep));
        }
      }
    }
  }

  Hx->initialize ();
  Hy->initialize ();
  Hz->initialize ();

  if (HxInitial != NULLPTR)
  {
    for (int i = 0; i < Hx->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hx->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hx->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hx->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (posAbs);

          Hx->getFieldPointValue (pos)->setCurValue (HxInitial (realCoord * gridStep, 1 * gridTimeStep));
        }
      }
    }
  }

  if (HyInitial != NULLPTR)
  {
    for (int i = 0; i < Hy->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hy->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hy->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hy->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (posAbs);

          Hy->getFieldPointValue (pos)->setCurValue (HyInitial (realCoord * gridStep, 1 * gridTimeStep));
        }
      }
    }
  }

  if (HzInitial != NULLPTR)
  {
    for (int i = 0; i < Hz->getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hz->getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hz->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = Hz->getTotalPosition (pos);
          GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (posAbs);

          Hz->getFieldPointValue (pos)->setCurValue (HzInitial (realCoord * gridStep, 1 * gridTimeStep));
        }
      }
    }
  }

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

    if (solverSettings.getDoUseTFSFPML ())
    {
      DInc->initialize ();
      BInc->initialize ();

      FPValue eps0 = PhysicsConst::Eps0;
      FPValue mu0 = PhysicsConst::Mu0;

      for (grid_coord i = 0; i < SigmaXInc->getSize ().getX (); ++i)
      {
        FieldPointValue* valSigma = new FieldPointValue ();

        GridCoordinate1D pos (i);
        GridCoordinate1D size = SigmaXInc->getSize ();

        /*
         * FIXME: add layout coordinates for material: sigma, eps, etc.
         */
        if (pos.getX () < solverSettings.getTFSFPMLSizeXLeft ())
        {
          FPValue boundary = solverSettings.getTFSFPMLSizeXLeft () * gridStep;
          uint32_t exponent = 8;
          FPValue R_err = 1e-16;
          FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
          FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

          grid_coord dist = solverSettings.getTFSFPMLSizeXLeft () - pos.getX ();
          FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
          FPValue x2 = dist * gridStep;       // lower bounds for point i

          FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));    //   polynomial grading

          valSigma->setCurValue (getFieldValueRealOnly (val));
        }
        else if (pos.getX () > size.getX () - solverSettings.getTFSFPMLSizeXRight ())
        {
          FPValue boundary = solverSettings.getTFSFPMLSizeXRight () * gridStep;
          uint32_t exponent = 8;
          FPValue R_err = 1e-16;
          FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
          FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

          grid_coord dist = pos.getX () - (size.getX () - solverSettings.getTFSFPMLSizeXRight ());
          FPValue x1 = (dist + 1) * gridStep;       // upper bounds for point i
          FPValue x2 = dist * gridStep;       // lower bounds for point i

          //std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
          FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));   //   polynomial grading

          valSigma->setCurValue (getFieldValueRealOnly (val));
        }

        valSigma->setCurValue (0);

        SigmaXInc->setFieldPointValue (valSigma, pos);
      }
    }
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());

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
                   Grid<GridCoordinate3D> *curEz,
                   Grid<GridCoordinate3D> *curHy,
                   Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHz1 = curHz->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHz2 = curHz->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHy1 = curHy->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHy2 = curHy->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHy1 != NULL && valHy2 != NULL);

      FieldValue Hz1 = valHz1->getCurValue ();
      FieldValue Hz2 = valHz2->getCurValue ();
      FieldValue Hy1 = valHy1->getCurValue ();
      FieldValue Hy2 = valHy2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos1));
        Hz2 -= yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos2));
        Hy1 -= yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos3));
        Hy2 -= yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos4));
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curEz,
                   Grid<GridCoordinate3D> *curHx,
                   Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHz1 = curHz->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHz2 = curHz->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHx1 = curHx->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHx2 = curHx->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hz1 = valHz1->getCurValue ();
      FieldValue Hz2 = valHz2->getCurValue ();
      FieldValue Hx1 = valHx1->getCurValue ();
      FieldValue Hx2 = valHx2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos1));
        Hz2 -= yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (pos2));
        Hx1 -= yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos3));
        Hx2 -= yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos4));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.getY ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.getY ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curEz,
                   Grid<GridCoordinate3D> *curHx,
                   Grid<GridCoordinate3D> *curHy)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHy1 = curHy->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHy2 = curHy->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHx1 = curHx->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHx2 = curHx->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHy1 != NULL && valHy2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hy1 = valHy1->getCurValue ();
      FieldValue Hy2 = valHy2->getCurValue ();
      FieldValue Hx1 = valHx1->getCurValue ();
      FieldValue Hx2 = valHx2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hy1 -= yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos1));
        Hy2 -= yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (pos2));
        Hx1 -= yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos3));
        Hx2 -= yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (pos4));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * (-(Hy1 + Hy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * ((Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curEy,
                   Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0.5,0,0));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0.5,0,0));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0.5,0,0));

      FieldPointValue *valEy11 = curEy->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEy12 = curEy->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEy21 = curEy->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEy22 = curEy->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEz11 = curEz->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEz12 = curEz->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEz21 = curEz->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEz22 = curEz->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ey1 = (valEy11->getCurValue () + valEy12->getCurValue ()) / FPValue(2.0);
      FieldValue Ey2 = (valEy21->getCurValue () + valEy22->getCurValue ()) / FPValue(2.0);
      FieldValue Ez1 = (valEz11->getCurValue () + valEz12->getCurValue ()) / FPValue(2.0);
      FieldValue Ez2 = (valEz21->getCurValue () + valEz22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ey1 -= yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos1));
        Ey2 -= yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos2));
        Ez1 -= yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos3));
        Ez2 -= yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos4));
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.getX ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curEx,
                   Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0,0.5,0));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0,0.5,0));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0,0.5,0));

      FieldPointValue *valEx11 = curEx->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEx12 = curEx->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEx21 = curEx->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEx22 = curEx->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEz11 = curEz->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEz12 = curEz->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEz21 = curEz->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEz22 = curEz->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ex1 = (valEx11->getCurValue () + valEx12->getCurValue ()) / FPValue(2.0);
      FieldValue Ex2 = (valEx21->getCurValue () + valEx22->getCurValue ()) / FPValue(2.0);
      FieldValue Ez1 = (valEz11->getCurValue () + valEz12->getCurValue ()) / FPValue(2.0);
      FieldValue Ez2 = (valEz21->getCurValue () + valEz22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos1));
        Ex2 -= yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos2));
        Ez1 -= yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos3));
        Ez2 -= yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (pos4));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.getY ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.getY ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffL_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                   Grid<GridCoordinate3D> *curEx,
                   Grid<GridCoordinate3D> *curEy,
                   Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().getX () / 2;
  FPValue diffy0 = curEz->getTotalSize ().getY () / 2;
  FPValue diffz0 = curEz->getTotalSize ().getZ () / 2;

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

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0,0,0.5));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0,0,0.5));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0,0,0.5));

      FieldPointValue *valEx11 = curEx->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEx12 = curEx->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEx21 = curEx->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEx22 = curEx->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEy11 = curEy->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEy12 = curEy->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEy21 = curEy->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEy22 = curEy->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos32)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL);

      FieldValue Ex1 = (valEx11->getCurValue () + valEx12->getCurValue ()) / FPValue(2.0);
      FieldValue Ex2 = (valEx21->getCurValue () + valEx22->getCurValue ()) / FPValue(2.0);
      FieldValue Ey1 = (valEy11->getCurValue () + valEy12->getCurValue ()) / FPValue(2.0);
      FieldValue Ey2 = (valEy21->getCurValue () + valEy22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos1));
        Ex2 -= yeeLayout->getExFromIncidentE (approximateIncidentWaveE (pos2));
        Ey1 -= yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos3));
        Ey2 -= yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (pos4));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * (-(Ey1 + Ey2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.getZ ()?1:-1) * ((Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return Scheme3D::NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

Scheme3D::NPair
Scheme3D::ntffN (FPValue angleTeta, FPValue anglePhi,
                 Grid<GridCoordinate3D> *curEz,
                 Grid<GridCoordinate3D> *curHx,
                 Grid<GridCoordinate3D> *curHy,
                 Grid<GridCoordinate3D> *curHz)
{
  return ntffN_x (leftNTFF.getX (), angleTeta, anglePhi, curEz, curHy, curHz)
         + ntffN_x (rightNTFF.getX (), angleTeta, anglePhi, curEz, curHy, curHz)
         + ntffN_y (leftNTFF.getY (), angleTeta, anglePhi, curEz, curHx, curHz)
         + ntffN_y (rightNTFF.getY (), angleTeta, anglePhi, curEz, curHx, curHz)
         + ntffN_z (leftNTFF.getZ (), angleTeta, anglePhi, curEz, curHx, curHy)
         + ntffN_z (rightNTFF.getZ (), angleTeta, anglePhi, curEz, curHx, curHy);
}

Scheme3D::NPair
Scheme3D::ntffL (FPValue angleTeta, FPValue anglePhi,
                 Grid<GridCoordinate3D> *curEx,
                 Grid<GridCoordinate3D> *curEy,
                 Grid<GridCoordinate3D> *curEz)
{
  return ntffL_x (leftNTFF.getX (), angleTeta, anglePhi, curEy, curEz)
         + ntffL_x (rightNTFF.getX (), angleTeta, anglePhi, curEy, curEz)
         + ntffL_y (leftNTFF.getY (), angleTeta, anglePhi, curEx, curEz)
         + ntffL_y (rightNTFF.getY (), angleTeta, anglePhi, curEx, curEz)
         + ntffL_z (leftNTFF.getZ (), angleTeta, anglePhi, curEx, curEy, curEz)
         + ntffL_z (rightNTFF.getZ (), angleTeta, anglePhi, curEx, curEy, curEz);
}

FPValue
Scheme3D::Pointing_scat (FPValue angleTeta, FPValue anglePhi,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

  NPair N = ntffN (angleTeta, anglePhi, curEz, curHx, curHy, curHz);
  NPair L = ntffL (angleTeta, anglePhi, curEx, curEy, curEz);

  int processId = 0;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();

    FieldValue tmpArray[4];
    FieldValue tmpArrayRes[4];
    const int count = 4;

    tmpArray[0] = N.nTeta;
    tmpArray[1] = N.nPhi;
    tmpArray[2] = L.nTeta;
    tmpArray[3] = L.nPhi;

    MPI_Datatype datatype;

#ifdef FLOAT_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_FLOAT;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_LONG_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_LONG_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* LONG_DOUBLE_VALUES */

    // gather all sum_teta and sum_phi on 0 node
    MPI_Reduce (tmpArray, tmpArrayRes, count, datatype, MPI_SUM, 0, ParallelGrid::getParallelCore ()->getCommunicator ());

    if (processId == 0)
    {
      N.nTeta = FieldValue (tmpArrayRes[0]);
      N.nPhi = FieldValue (tmpArrayRes[1]);

      L.nTeta = FieldValue (tmpArrayRes[2]);
      L.nPhi = FieldValue (tmpArrayRes[3]);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  if (processId == 0)
  {
    FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);

    FieldValue first = -L.nPhi + n0 * N.nTeta;
    FieldValue second = -L.nTeta - n0 * N.nPhi;

    FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
    FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());

    return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
  }
  else
  {
    return 0.0;
  }
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

FPValue
Scheme3D::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
{
  return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0);
}

void
Scheme3D::makeGridScattered (Grid<GridCoordinate3D> *grid, GridType gridType)
{
  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
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

  GridCoordinate3D startEx;
  GridCoordinate3D endEx;
  GridCoordinate3D startEy;
  GridCoordinate3D endEy;
  GridCoordinate3D startEz;
  GridCoordinate3D endEz;
  GridCoordinate3D startHx;
  GridCoordinate3D endHx;
  GridCoordinate3D startHy;
  GridCoordinate3D endHy;
  GridCoordinate3D startHz;
  GridCoordinate3D endHz;

  if (solverSettings.getDoUseManualStartEndDumpCoord ())
  {
    GridCoordinate3D start (solverSettings.getSaveStartCoordX (),
                            solverSettings.getSaveStartCoordY (),
                            solverSettings.getSaveStartCoordZ ());
    GridCoordinate3D end (solverSettings.getSaveEndCoordX (),
                          solverSettings.getSaveEndCoordY (),
                          solverSettings.getSaveEndCoordZ ());

    startEx = startEy = startEz = startHx = startHy = startHz = start;
    endEx = endEy = endEz = endHx = endHy = endHz = end;
  }
  else
  {
    startEx = getStartCoord (GridType::EX, Ex->getTotalSize ());
    endEx = getEndCoord (GridType::EX, Ex->getTotalSize ());

    startEy = getStartCoord (GridType::EY, Ey->getTotalSize ());
    endEy = getEndCoord (GridType::EY, Ey->getTotalSize ());

    startEz = getStartCoord (GridType::EZ, Ez->getTotalSize ());
    endEz = getEndCoord (GridType::EZ, Ez->getTotalSize ());

    startHx = getStartCoord (GridType::HX, Hx->getTotalSize ());
    endHx = getEndCoord (GridType::HX, Hx->getTotalSize ());

    startHy = getStartCoord (GridType::HY, Hy->getTotalSize ());
    endHy = getEndCoord (GridType::HY, Hy->getTotalSize ());

    startHz = getStartCoord (GridType::HZ, Hz->getTotalSize ());
    endHz = getEndCoord (GridType::HZ, Hz->getTotalSize ());
  }

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

    if (solverSettings.getDoSaveTFSFEInc ()
        || solverSettings.getDoSaveTFSFHInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->init (t, CURRENT, processId, "EInc");
      dumper1D[type]->dumpGrid (EInc, GridCoordinate1D (0), EInc->getSize ());

      dumper1D[type]->init (t, CURRENT, processId, "HInc");
      dumper1D[type]->dumpGrid (HInc, GridCoordinate1D (0), HInc->getSize ());
    }
  }
}

void
Scheme3D::saveNTFF (bool isReverse, time_step t)
{
  int processId = 0;

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  std::ofstream outfile;
  std::ostream *outs;
  const char *strName;
  FPValue start;
  FPValue end;
  FPValue step;

  if (isReverse)
  {
    strName = "Reverse diagram";
    start = yeeLayout->getIncidentWaveAngle2 ();
    end = yeeLayout->getIncidentWaveAngle2 ();
    step = 1.0;
  }
  else
  {
    strName = "Forward diagram";
    start = 0.0;
    end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
    step = PhysicsConst::Pi * solverSettings.getAngleStepNTFF () / 180;
  }

  if (processId == 0)
  {
    if (solverSettings.getDoSaveNTFFToStdout ())
    {
      outs = &std::cout;
    }
    else
    {
      outfile.open (solverSettings.getFileNameNTFF ().c_str ());
      outs = &outfile;
    }
    (*outs) << strName << std::endl << std::endl;
  }

  for (FPValue angle = start; angle <= end; angle += step)
  {
    FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
                                 angle,
                                 Ex,
                                 Ey,
                                 Ez,
                                 Hx,
                                 Hy,
                                 Hz) / Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);

    if (processId == 0)
    {
      (*outs) << "timestep = "
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
  }

  if (processId == 0)
  {
    if (!solverSettings.getDoSaveNTFFToStdout ())
    {
      outfile.close ();
    }
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

GridCoordinate3D
Scheme3D::getStartCoord (GridType gridType, GridCoordinate3D size)
{
  GridCoordinate3D start (0, 0, 0);
  if (solverSettings.getDoSaveWithoutPML ()
      && solverSettings.getDoUsePML ())
  {
    GridCoordinate3D leftBorder = yeeLayout->getLeftBorderPML ();
    GridCoordinateFP3D min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    start.setX (leftBorder.getX () - min.getX () + 1);
    start.setY (leftBorder.getY () - min.getY () + 1);
    start.setZ (leftBorder.getZ () - min.getZ () + 1);
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
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

  if (orthogonalAxis == OrthogonalAxis::Z)
  {
    return GridCoordinate3D (start.getX (), start.getY (), size.getZ () / 2);
  }
  else if (orthogonalAxis == OrthogonalAxis::Y)
  {
    return GridCoordinate3D (start.getX (), size.getY () / 2, start.getZ ());
  }
  else if (orthogonalAxis == OrthogonalAxis::X)
  {
    return GridCoordinate3D (size.getX () / 2, start.getY (), start.getZ ());
  }
}

GridCoordinate3D
Scheme3D::getEndCoord (GridType gridType, GridCoordinate3D size)
{
  GridCoordinate3D end = size;
  if (solverSettings.getDoSaveWithoutPML ()
      && solverSettings.getDoUsePML ())
  {
    GridCoordinate3D rightBorder = yeeLayout->getRightBorderPML ();
    GridCoordinateFP3D min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    end.setX (rightBorder.getX () - min.getX ());
    end.setY (rightBorder.getY () - min.getY ());
    end.setZ (rightBorder.getZ () - min.getZ ());
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
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

  if (orthogonalAxis == OrthogonalAxis::Z)
  {
    return GridCoordinate3D (end.getX (), end.getY (), size.getZ () / 2 + 1);
  }
  else if (orthogonalAxis == OrthogonalAxis::Y)
  {
    return GridCoordinate3D (end.getX (), size.getY () / 2 + 1, end.getZ ());
  }
  else if (orthogonalAxis == OrthogonalAxis::X)
  {
    return GridCoordinate3D (size.getX () / 2 + 1, end.getY (), end.getZ ());
  }
}

#endif /* GRID_3D */

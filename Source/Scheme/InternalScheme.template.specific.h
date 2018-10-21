template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
InternalSchemeBase<Type, TCoord, layout_type>::~InternalSchemeBase ()
{
  delete Eps;
  delete Mu;

  delete Ex;
  delete Ey;
  delete Ez;

  delete Hx;
  delete Hy;
  delete Hz;

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    delete Dx;
    delete Dy;
    delete Dz;

    delete Bx;
    delete By;
    delete Bz;

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
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

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    delete ExAmplitude;
    delete EyAmplitude;
    delete EzAmplitude;
    delete HxAmplitude;
    delete HyAmplitude;
    delete HzAmplitude;
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    delete OmegaPE;
    delete OmegaPM;
    delete GammaE;
    delete GammaM;
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    delete EInc;
    delete HInc;
  }
}

#if defined (PARALLEL_GRID)

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalScheme1D<Type, layout_type>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               ct1, ct2, ct3);

  InternalSchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalScheme2D<Type, layout_type>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               ct1, ct2, ct3);

  InternalSchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalScheme2D<Type, layout_type>::allocateParallelGrids ()
{
#ifdef GRID_3D
  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               ct1, ct2, ct3);

  InternalSchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=3.");
#endif
}

#endif /* PARALLEL_GRID */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type>::init (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                                           bool parallel)
{
  yeeLayout = layout;
  useParallel = parallel;

  initCoordTypes ();

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = layout->getEzSize () - leftNTFF + TC (1, 1, 1
#ifdef DEBUG_INFO
                                                      , ct1, ct2, ct3
#endif
                                                      );
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
    allocateParallelGrids ();
#else
    ALWAYS_ASSERT (false);
#endif
  }
  else
  {
    allocateGrids ();
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    allocateGridsInc ();
  }

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateGrids (InternalSchemeBase<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  intScheme->Eps = new Grid< TCoord<grid_coord, true> > (layout->getEpsSize (), 0, "Eps");
  intScheme->Mu = new Grid<TC> (layout->getEpsSize (), 0, "Mu");

  intScheme->Ex = intScheme->doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "Ex") : NULLPTR;
  intScheme->Ey = intScheme->doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "Ey") : NULLPTR;
  intScheme->Ez = intScheme->doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "Ez") : NULLPTR;
  intScheme->Hx = intScheme->doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "Hx") : NULLPTR;
  intScheme->Hy = intScheme->doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "Hy") : NULLPTR;
  intScheme->Hz = intScheme->doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "Hz") : NULLPTR;

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    intScheme->Dx = intScheme->doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "Dx") : NULLPTR;
    intScheme->Dy = intScheme->doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "Dy") : NULLPTR;
    intScheme->Dz = intScheme->doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "Dz") : NULLPTR;
    intScheme->Bx = intScheme->doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "Bx") : NULLPTR;
    intScheme->By = intScheme->doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "By") : NULLPTR;
    intScheme->Bz = intScheme->doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "Bz") : NULLPTR;

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      intScheme->D1x = intScheme->doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "D1x") : NULLPTR;
      intScheme->D1y = intScheme->doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "D1y") : NULLPTR;
      intScheme->D1z = intScheme->doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "D1z") : NULLPTR;
      intScheme->B1x = intScheme->doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "B1x") : NULLPTR;
      intScheme->B1y = intScheme->doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "B1y") : NULLPTR;
      intScheme->B1z = intScheme->doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "B1z") : NULLPTR;
    }

    intScheme->SigmaX = intScheme->doNeedSigmaX ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaX") : NULLPTR;
    intScheme->SigmaY = intScheme->doNeedSigmaY ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaY") : NULLPTR;
    intScheme->SigmaZ = intScheme->doNeedSigmaZ ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaZ") : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    intScheme->ExAmplitude = intScheme->doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "ExAmp") : NULLPTR;
    intScheme->EyAmplitude = intScheme->doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "EyAmp") : NULLPTR;
    intScheme->EzAmplitude = intScheme->doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "EzAmp") : NULLPTR;
    intScheme->HxAmplitude = intScheme->doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "HxAmp") : NULLPTR;
    intScheme->HyAmplitude = intScheme->doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "HyAmp") : NULLPTR;
    intScheme->HzAmplitude = intScheme->doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "HzAmp") : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    intScheme->OmegaPE = new Grid<TC> (layout->getEpsSize (), 0, "OmegaPE");
    intScheme->GammaE = new Grid<TC> (layout->getEpsSize (), 0, "GammaE");
    intScheme->OmegaPM = new Grid<TC> (layout->getEpsSize (), 0, "OmegaPM");
    intScheme->GammaM = new Grid<TC> (layout->getEpsSize (), 0, "GammaM");
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateGridsInc (InternalSchemeBase<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
{
  intScheme->EInc = new Grid<GridCoordinate1D> (GridCoordinate1D (500*(layout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
#endif
                                                            ), 0, "EInc");
  intScheme->HInc = new Grid<GridCoordinate1D> (GridCoordinate1D (500*(layout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
#endif
                                                            ), 0, "HInc");
}

#ifdef PARALLEL_GRID

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateParallelGrids (ParallelYeeGridLayout<Type, layout_type> *pLayout,
                            bool doNeedEx, bool doNeedEy, bool doNeedEz,
                            bool doNeedHx, bool doNeedHy, bool doNeedHz,
                            bool doNeedSigmaX, bool doNeedSigmaY, bool doNeedSigmaZ,
                            ParallelGridCoordinate bufSize, ParallelGrid **Eps, ParallelGrid **Mu,
                            ParallelGrid **Ex, ParallelGrid **Ey, ParallelGrid **Ez,
                            ParallelGrid **Hx, ParallelGrid **Hy, ParallelGrid **Hz,
                            ParallelGrid **Dx, ParallelGrid **Dy, ParallelGrid **Dz,
                            ParallelGrid **Bx, ParallelGrid **By, ParallelGrid **Bz,
                            ParallelGrid **D1x, ParallelGrid **D1y, ParallelGrid **D1z,
                            ParallelGrid **B1x, ParallelGrid **B1y, ParallelGrid **B1z,
                            ParallelGrid **SigmaX, ParallelGrid **SigmaY, ParallelGrid **SigmaZ,
                            ParallelGrid **ExAmplitude, ParallelGrid **EyAmplitude, ParallelGrid **EzAmplitude,
                            ParallelGrid **HxAmplitude, ParallelGrid **HyAmplitude, ParallelGrid **HzAmplitude,
                            ParallelGrid **OmegaPE, ParallelGrid **GammaE,
                            ParallelGrid **OmegaPM, ParallelGrid **GammaM)
{
  *Eps = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "Eps");
  *Mu = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getMuSizeForCurNode (), "Mu");

  *Ex = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "Ex") : NULLPTR;
  *Ey = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "Ey") : NULLPTR;
  *Ez = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "Ez") : NULLPTR;
  *Hx = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "Hx") : NULLPTR;
  *Hy = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "Hy") : NULLPTR;
  *Hz = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "Hz") : NULLPTR;

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    *Dx = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "Dx") : NULLPTR;
    *Dy = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "Dy") : NULLPTR;
    *Dz = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "Dz") : NULLPTR;
    *Bx = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "Bx") : NULLPTR;
    *By = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "By") : NULLPTR;
    *Bz = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "Bz") : NULLPTR;

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      *D1x = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "D1x") : NULLPTR;
      *D1y = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "D1y") : NULLPTR;
      *D1z = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "D1z") : NULLPTR;
      *B1x = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "B1x") : NULLPTR;
      *B1y = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "B1y") : NULLPTR;
      *B1z = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "B1z") : NULLPTR;
    }

    *SigmaX = doNeedSigmaX ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaX") : NULLPTR;
    *SigmaY = doNeedSigmaY ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaY") : NULLPTR;
    *SigmaZ = doNeedSigmaZ ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaZ") : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    *ExAmplitude = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "ExAmp") : NULLPTR;
    *EyAmplitude = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "EyAmp") : NULLPTR;
    *EzAmplitude = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "EzAmp") : NULLPTR;
    *HxAmplitude = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "HxAmp") : NULLPTR;
    *HyAmplitude = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "HyAmp") : NULLPTR;
    *HzAmplitude = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "HzAmp") : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    *OmegaPE = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "OmegaPE");
    *GammaE = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "GammaE");
    *OmegaPM = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "OmegaPM");
    *GammaM = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "GammaM");
  }
}

#endif /* PARALLEL_GRID */

#ifdef CUDA_ENABLED

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::allocateGridsFromCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *intScheme,
                                               InternalSchemeBase<Type, TCoord, layout_type> *cpuScheme, TCoord<grid_coord, true> blockSize, TCoord<grid_coord, true> bufSize)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  intScheme->Eps = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Eps);
  intScheme->Mu = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Mu);

  intScheme->Ex = intScheme->doNeedEx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Ex) : NULLPTR;
  intScheme->Ey = intScheme->doNeedEy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Ey) : NULLPTR;
  intScheme->Ez = intScheme->doNeedEz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Ez) : NULLPTR;
  intScheme->Hx = intScheme->doNeedHx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Hx) : NULLPTR;
  intScheme->Hy = intScheme->doNeedHy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Hy) : NULLPTR;
  intScheme->Hz = intScheme->doNeedHz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Hz) : NULLPTR;

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    intScheme->Dx = intScheme->doNeedEx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Dx) : NULLPTR;
    intScheme->Dy = intScheme->doNeedEy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Dy) : NULLPTR;
    intScheme->Dz = intScheme->doNeedEz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Dz) : NULLPTR;
    intScheme->Bx = intScheme->doNeedHx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Bx) : NULLPTR;
    intScheme->By = intScheme->doNeedHy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->By) : NULLPTR;
    intScheme->Bz = intScheme->doNeedHz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->Bz) : NULLPTR;

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      intScheme->D1x = intScheme->doNeedEx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->D1x) : NULLPTR;
      intScheme->D1y = intScheme->doNeedEy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->D1y) : NULLPTR;
      intScheme->D1z = intScheme->doNeedEz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->D1z) : NULLPTR;
      intScheme->B1x = intScheme->doNeedHx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->B1x) : NULLPTR;
      intScheme->B1y = intScheme->doNeedHy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->B1y) : NULLPTR;
      intScheme->B1z = intScheme->doNeedHz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->B1z) : NULLPTR;
    }

    intScheme->SigmaX = intScheme->doNeedSigmaX ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->SigmaX) : NULLPTR;
    intScheme->SigmaY = intScheme->doNeedSigmaY ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->SigmaY) : NULLPTR;
    intScheme->SigmaZ = intScheme->doNeedSigmaZ ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->SigmaZ) : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    intScheme->ExAmplitude = intScheme->doNeedEx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->ExAmplitude) : NULLPTR;
    intScheme->EyAmplitude = intScheme->doNeedEy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->EyAmplitude) : NULLPTR;
    intScheme->EzAmplitude = intScheme->doNeedEz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->EzAmplitude) : NULLPTR;
    intScheme->HxAmplitude = intScheme->doNeedHx ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->HxAmplitude) : NULLPTR;
    intScheme->HyAmplitude = intScheme->doNeedHy ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->HyAmplitude) : NULLPTR;
    intScheme->HzAmplitude = intScheme->doNeedHz ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->HzAmplitude) : NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    intScheme->OmegaPE = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->OmegaPE);
    intScheme->GammaE = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->GammaE);
    intScheme->OmegaPM = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->OmegaPM);
    intScheme->GammaM = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->GammaM);
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    TC one (1
#ifdef DEBUG_INFO
            , CoordinateType::X
#endif
            );

    intScheme->EInc = new CudaGrid<GridCoordinate1D> (GridCoordinate1D (500*(cpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                             ), one, cpuScheme->EInc);
    intScheme->HInc = new CudaGrid<GridCoordinate1D> (GridCoordinate1D (500*(cpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                             ), one, cpuScheme->HInc);
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
InternalSchemeBaseGPU<Type, TCoord, layout_type>::~InternalSchemeBaseGPU ()
{
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::allocateGridsOnGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Eps, sizeof(CudaGrid<TC>)));
  cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Mu, sizeof(CudaGrid<TC>)));

  if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ex, sizeof(CudaGrid<TC>))); }
  if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ey, sizeof(CudaGrid<TC>))); }
  if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ez, sizeof(CudaGrid<TC>))); }
  if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hx, sizeof(CudaGrid<TC>))); }
  if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hy, sizeof(CudaGrid<TC>))); }
  if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hz, sizeof(CudaGrid<TC>))); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dx, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dy, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dz, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Bx, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->By, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Bz, sizeof(CudaGrid<TC>))); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1x, sizeof(CudaGrid<TC>))); }
      if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1y, sizeof(CudaGrid<TC>))); }
      if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1z, sizeof(CudaGrid<TC>))); }
      if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1x, sizeof(CudaGrid<TC>))); }
      if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1y, sizeof(CudaGrid<TC>))); }
      if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1z, sizeof(CudaGrid<TC>))); }
    }

    if (gpuScheme->doNeedSigmaX) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaX, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedSigmaY) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaY, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedSigmaZ) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaZ, sizeof(CudaGrid<TC>))); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->ExAmplitude, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EyAmplitude, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EzAmplitude, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HxAmplitude, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HyAmplitude, sizeof(CudaGrid<TC>))); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HzAmplitude, sizeof(CudaGrid<TC>))); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->OmegaPE, sizeof(CudaGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->GammaE, sizeof(CudaGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->OmegaPM, sizeof(CudaGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->GammaM, sizeof(CudaGrid<TC>)));
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EInc, sizeof(CudaGrid<GridCoordinate1D>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HInc, sizeof(CudaGrid<GridCoordinate1D>)));
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsFromCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme,
                  TCoord<grid_coord, true> start,
                  TCoord<grid_coord, true> end)
{
  gpuScheme->Eps->copyFromCPU (start, end);
  gpuScheme->Mu->copyFromCPU (start, end);

  if (gpuScheme->doNeedEx) { gpuScheme->Ex->copyFromCPU (start, end); }
  if (gpuScheme->doNeedEy) { gpuScheme->Ey->copyFromCPU (start, end); }
  if (gpuScheme->doNeedEz) { gpuScheme->Ez->copyFromCPU (start, end); }
  if (gpuScheme->doNeedHx) { gpuScheme->Hx->copyFromCPU (start, end); }
  if (gpuScheme->doNeedHy) { gpuScheme->Hy->copyFromCPU (start, end); }
  if (gpuScheme->doNeedHz) { gpuScheme->Hz->copyFromCPU (start, end); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { gpuScheme->Dx->copyFromCPU (start, end); }
    if (gpuScheme->doNeedEy) { gpuScheme->Dy->copyFromCPU (start, end); }
    if (gpuScheme->doNeedEz) { gpuScheme->Dz->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHx) { gpuScheme->Bx->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHy) { gpuScheme->By->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHz) { gpuScheme->Bz->copyFromCPU (start, end); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { gpuScheme->D1x->copyFromCPU (start, end); }
      if (gpuScheme->doNeedEy) { gpuScheme->D1y->copyFromCPU (start, end); }
      if (gpuScheme->doNeedEz) { gpuScheme->D1z->copyFromCPU (start, end); }
      if (gpuScheme->doNeedHx) { gpuScheme->B1x->copyFromCPU (start, end); }
      if (gpuScheme->doNeedHy) { gpuScheme->B1y->copyFromCPU (start, end); }
      if (gpuScheme->doNeedHz) { gpuScheme->B1z->copyFromCPU (start, end); }
    }

    if (gpuScheme->doNeedSigmaX) { gpuScheme->SigmaX->copyFromCPU (start, end); }
    if (gpuScheme->doNeedSigmaY) { gpuScheme->SigmaY->copyFromCPU (start, end); }
    if (gpuScheme->doNeedSigmaZ) { gpuScheme->SigmaZ->copyFromCPU (start, end); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { gpuScheme->ExAmplitude->copyFromCPU (start, end); }
    if (gpuScheme->doNeedEy) { gpuScheme->EyAmplitude->copyFromCPU (start, end); }
    if (gpuScheme->doNeedEz) { gpuScheme->EzAmplitude->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHx) { gpuScheme->HxAmplitude->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHy) { gpuScheme->HyAmplitude->copyFromCPU (start, end); }
    if (gpuScheme->doNeedHz) { gpuScheme->HzAmplitude->copyFromCPU (start, end); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    gpuScheme->OmegaPE->copyFromCPU (start, end);
    gpuScheme->GammaE->copyFromCPU (start, end);
    gpuScheme->OmegaPM->copyFromCPU (start, end);
    gpuScheme->GammaM->copyFromCPU (start, end);
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    TCoord<grid_coord, true> zero (0
#ifdef DEBUG_INFO
            , CoordinateType::X
#endif
            );



    gpuScheme->EInc->copyFromCPU (zero, GridCoordinate1D (500*(gpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                             ));
    gpuScheme->HInc->copyFromCPU (zero, GridCoordinate1D (500*(gpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                             ));
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsToGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme,
                                         InternalSchemeBaseGPU<Type, TCoord, layout_type> *intScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Eps, intScheme->Eps, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Mu, intScheme->Mu, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));

  if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ex, intScheme->Ex, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ey, intScheme->Ey, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ez, intScheme->Ez, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hx, intScheme->Hx, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hy, intScheme->Hy, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hz, intScheme->Hz, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dx, intScheme->Dx, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dy, intScheme->Dy, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dz, intScheme->Dz, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Bx, intScheme->Bx, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->By, intScheme->By, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Bz, intScheme->Bz, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1x, intScheme->D1x, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1y, intScheme->D1y, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1z, intScheme->D1z, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1x, intScheme->B1x, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1y, intScheme->B1y, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1z, intScheme->B1z, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    }

    if (gpuScheme->doNeedSigmaX) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaX, intScheme->SigmaX, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedSigmaY) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaY, intScheme->SigmaY, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedSigmaZ) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaZ, intScheme->SigmaZ, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->ExAmplitude, intScheme->ExAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EyAmplitude, intScheme->EyAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EzAmplitude, intScheme->EzAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HxAmplitude, intScheme->HxAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HyAmplitude, intScheme->HyAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HzAmplitude, intScheme->HzAmplitude, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->OmegaPE, intScheme->OmegaPE, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->GammaE, intScheme->GammaE, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->OmegaPM, intScheme->OmegaPM, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->GammaM, intScheme->GammaM, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EInc, intScheme->EInc, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HInc, intScheme->HInc, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsBackToCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme)
{
  gpuScheme->Eps->copyToCPU ();
  gpuScheme->Mu->copyToCPU ();

  if (gpuScheme->doNeedEx) { gpuScheme->Ex->copyToCPU (); }
  if (gpuScheme->doNeedEy) { gpuScheme->Ey->copyToCPU (); }
  if (gpuScheme->doNeedEz) { gpuScheme->Ez->copyToCPU (); }
  if (gpuScheme->doNeedHx) { gpuScheme->Hx->copyToCPU (); }
  if (gpuScheme->doNeedHy) { gpuScheme->Hy->copyToCPU (); }
  if (gpuScheme->doNeedHz) { gpuScheme->Hz->copyToCPU (); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { gpuScheme->Dx->copyToCPU (); }
    if (gpuScheme->doNeedEy) { gpuScheme->Dy->copyToCPU (); }
    if (gpuScheme->doNeedEz) { gpuScheme->Dz->copyToCPU (); }
    if (gpuScheme->doNeedHx) { gpuScheme->Bx->copyToCPU (); }
    if (gpuScheme->doNeedHy) { gpuScheme->By->copyToCPU (); }
    if (gpuScheme->doNeedHz) { gpuScheme->Bz->copyToCPU (); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { gpuScheme->D1x->copyToCPU (); }
      if (gpuScheme->doNeedEy) { gpuScheme->D1y->copyToCPU (); }
      if (gpuScheme->doNeedEz) { gpuScheme->D1z->copyToCPU (); }
      if (gpuScheme->doNeedHx) { gpuScheme->B1x->copyToCPU (); }
      if (gpuScheme->doNeedHy) { gpuScheme->B1y->copyToCPU (); }
      if (gpuScheme->doNeedHz) { gpuScheme->B1z->copyToCPU (); }
    }

    if (gpuScheme->doNeedSigmaX) { gpuScheme->SigmaX->copyToCPU (); }
    if (gpuScheme->doNeedSigmaY) { gpuScheme->SigmaY->copyToCPU (); }
    if (gpuScheme->doNeedSigmaZ) { gpuScheme->SigmaZ->copyToCPU (); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { gpuScheme->ExAmplitude->copyToCPU (); }
    if (gpuScheme->doNeedEy) { gpuScheme->EyAmplitude->copyToCPU (); }
    if (gpuScheme->doNeedEz) { gpuScheme->EzAmplitude->copyToCPU (); }
    if (gpuScheme->doNeedHx) { gpuScheme->HxAmplitude->copyToCPU (); }
    if (gpuScheme->doNeedHy) { gpuScheme->HyAmplitude->copyToCPU (); }
    if (gpuScheme->doNeedHz) { gpuScheme->HzAmplitude->copyToCPU (); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    gpuScheme->OmegaPE->copyToCPU ();
    gpuScheme->GammaE->copyToCPU ();
    gpuScheme->OmegaPM->copyToCPU ();
    gpuScheme->GammaM->copyToCPU ();
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    gpuScheme->EInc->copyToCPU ();
    gpuScheme->HInc->copyToCPU ();
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBaseGPU<Type, TCoord, layout_type>::initFromCPU (InternalSchemeBase<Type, TCoord, layout_type> *cpuScheme,
                                                                  TC blockSize,
                                                                  TC bufSize)
{
  ASSERT (cpuScheme->isInitialized);

  yeeLayout = cpuScheme->yeeLayout;
  initScheme (cpuScheme->gridStep, cpuScheme->sourceWaveLength);

  useParallel = false;

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = cpuScheme->yeeLayout->getEzSize () - leftNTFF + one;
  }

  allocateGridsFromCPU (cpuScheme, blockSize, bufSize);

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBaseGPU<Type, TCoord, layout_type>::initOnGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme)
{
  yeeLayout = gpuScheme->yeeLayout;
  initScheme (gpuScheme->gridStep, gpuScheme->sourceWaveLength);

  useParallel = false;

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = gpuScheme->yeeLayout->getEzSize () - leftNTFF + one;
  }

  cudaCheckErrorCmd (cudaMalloc ((void **) &yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>)));
  cudaCheckErrorCmd (cudaMemcpy (yeeLayout, gpuScheme->yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));

  allocateGridsOnGPU ();

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBaseGPU<Type, TCoord, layout_type>::copyFromCPU (TCoord<grid_coord, true> start,
                                                                   TCoord<grid_coord, true> end)
{
  ASSERT (isInitialized);

  copyGridsFromCPU (start, end);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBaseGPU<Type, TCoord, layout_type>::copyToGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme)
{
  ASSERT (isInitialized);
  ASSERT (gpuScheme->isInitialized);

  copyGridsToGPU (gpuScheme);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeBaseGPU<Type, TCoord, layout_type>::copyBackToCPU ()
{
  ASSERT (isInitialized);

  copyGridsBackToCPU ();
}

#endif /* CUDA_ENABLED */
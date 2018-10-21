#ifndef INTERNAL_SCHEME_HELPER_H
#define INTERNAL_SCHEME_HELPER_H

#include "GridInterface.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
class InternalSchemeBase;

class InternalSchemeHelper
{
public:

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
  CUDA_HOST
  static void allocateGrids (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
  {
    typedef TCoord<grid_coord, true> TC;
    typedef TCoord<grid_coord, false> TCS;
    typedef TCoord<FPValue, true> TCFP;
    typedef TCoord<FPValue, false> TCSFP;

    intScheme->Eps = new TGrid< TCoord<grid_coord, true> > (layout->getEpsSize (), 0, "Eps");
    intScheme->Mu = new TGrid<TC> (layout->getEpsSize (), 0, "Mu");

    intScheme->Ex = intScheme->doNeedEx ? new TGrid<TC> (layout->getExSize (), 0, "Ex") : NULLPTR;
    intScheme->Ey = intScheme->doNeedEy ? new TGrid<TC> (layout->getEySize (), 0, "Ey") : NULLPTR;
    intScheme->Ez = intScheme->doNeedEz ? new TGrid<TC> (layout->getEzSize (), 0, "Ez") : NULLPTR;
    intScheme->Hx = intScheme->doNeedHx ? new TGrid<TC> (layout->getHxSize (), 0, "Hx") : NULLPTR;
    intScheme->Hy = intScheme->doNeedHy ? new TGrid<TC> (layout->getHySize (), 0, "Hy") : NULLPTR;
    intScheme->Hz = intScheme->doNeedHz ? new TGrid<TC> (layout->getHzSize (), 0, "Hz") : NULLPTR;

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      intScheme->Dx = intScheme->doNeedEx ? new TGrid<TC> (layout->getExSize (), 0, "Dx") : NULLPTR;
      intScheme->Dy = intScheme->doNeedEy ? new TGrid<TC> (layout->getEySize (), 0, "Dy") : NULLPTR;
      intScheme->Dz = intScheme->doNeedEz ? new TGrid<TC> (layout->getEzSize (), 0, "Dz") : NULLPTR;
      intScheme->Bx = intScheme->doNeedHx ? new TGrid<TC> (layout->getHxSize (), 0, "Bx") : NULLPTR;
      intScheme->By = intScheme->doNeedHy ? new TGrid<TC> (layout->getHySize (), 0, "By") : NULLPTR;
      intScheme->Bz = intScheme->doNeedHz ? new TGrid<TC> (layout->getHzSize (), 0, "Bz") : NULLPTR;

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->D1x = intScheme->doNeedEx ? new TGrid<TC> (layout->getExSize (), 0, "D1x") : NULLPTR;
        intScheme->D1y = intScheme->doNeedEy ? new TGrid<TC> (layout->getEySize (), 0, "D1y") : NULLPTR;
        intScheme->D1z = intScheme->doNeedEz ? new TGrid<TC> (layout->getEzSize (), 0, "D1z") : NULLPTR;
        intScheme->B1x = intScheme->doNeedHx ? new TGrid<TC> (layout->getHxSize (), 0, "B1x") : NULLPTR;
        intScheme->B1y = intScheme->doNeedHy ? new TGrid<TC> (layout->getHySize (), 0, "B1y") : NULLPTR;
        intScheme->B1z = intScheme->doNeedHz ? new TGrid<TC> (layout->getHzSize (), 0, "B1z") : NULLPTR;
      }

      intScheme->SigmaX = intScheme->doNeedSigmaX ? new TGrid<TC> (layout->getEpsSize (), 0, "SigmaX") : NULLPTR;
      intScheme->SigmaY = intScheme->doNeedSigmaY ? new TGrid<TC> (layout->getEpsSize (), 0, "SigmaY") : NULLPTR;
      intScheme->SigmaZ = intScheme->doNeedSigmaZ ? new TGrid<TC> (layout->getEpsSize (), 0, "SigmaZ") : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      intScheme->ExAmplitude = intScheme->doNeedEx ? new TGrid<TC> (layout->getExSize (), 0, "ExAmp") : NULLPTR;
      intScheme->EyAmplitude = intScheme->doNeedEy ? new TGrid<TC> (layout->getEySize (), 0, "EyAmp") : NULLPTR;
      intScheme->EzAmplitude = intScheme->doNeedEz ? new TGrid<TC> (layout->getEzSize (), 0, "EzAmp") : NULLPTR;
      intScheme->HxAmplitude = intScheme->doNeedHx ? new TGrid<TC> (layout->getHxSize (), 0, "HxAmp") : NULLPTR;
      intScheme->HyAmplitude = intScheme->doNeedHy ? new TGrid<TC> (layout->getHySize (), 0, "HyAmp") : NULLPTR;
      intScheme->HzAmplitude = intScheme->doNeedHz ? new TGrid<TC> (layout->getHzSize (), 0, "HzAmp") : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      intScheme->OmegaPE = new TGrid<TC> (layout->getEpsSize (), 0, "OmegaPE");
      intScheme->GammaE = new TGrid<TC> (layout->getEpsSize (), 0, "GammaE");
      intScheme->OmegaPM = new TGrid<TC> (layout->getEpsSize (), 0, "OmegaPM");
      intScheme->GammaM = new TGrid<TC> (layout->getEpsSize (), 0, "GammaM");
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
  CUDA_HOST
  static void allocateGridsInc (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
  {
    intScheme->EInc = new TGrid<GridCoordinate1D> (GridCoordinate1D (500*(layout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                              ), 0, "EInc");
    intScheme->HInc = new TGrid<GridCoordinate1D> (GridCoordinate1D (500*(layout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                              ), 0, "HInc");
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid1, template <typename> class TGrid2>
  CUDA_HOST
  static void
  allocateGridsFromCPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid1> *intScheme,
                        InternalSchemeBase<Type, TCoord, layout_type, TGrid2> *cpuScheme, TCoord<grid_coord, true> blockSize, TCoord<grid_coord, true> bufSize)
  {
    typedef TCoord<grid_coord, true> TC;
    typedef TCoord<grid_coord, false> TCS;
    typedef TCoord<FPValue, true> TCFP;
    typedef TCoord<FPValue, false> TCSFP;

    intScheme->Eps = new TGrid1<TC> (blockSize, bufSize, cpuScheme->Eps);
    intScheme->Mu = new TGrid1<TC> (blockSize, bufSize, cpuScheme->Mu);

    intScheme->Ex = intScheme->doNeedEx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Ex) : NULLPTR;
    intScheme->Ey = intScheme->doNeedEy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Ey) : NULLPTR;
    intScheme->Ez = intScheme->doNeedEz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Ez) : NULLPTR;
    intScheme->Hx = intScheme->doNeedHx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Hx) : NULLPTR;
    intScheme->Hy = intScheme->doNeedHy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Hy) : NULLPTR;
    intScheme->Hz = intScheme->doNeedHz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Hz) : NULLPTR;

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      intScheme->Dx = intScheme->doNeedEx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Dx) : NULLPTR;
      intScheme->Dy = intScheme->doNeedEy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Dy) : NULLPTR;
      intScheme->Dz = intScheme->doNeedEz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Dz) : NULLPTR;
      intScheme->Bx = intScheme->doNeedHx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Bx) : NULLPTR;
      intScheme->By = intScheme->doNeedHy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->By) : NULLPTR;
      intScheme->Bz = intScheme->doNeedHz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->Bz) : NULLPTR;

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->D1x = intScheme->doNeedEx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->D1x) : NULLPTR;
        intScheme->D1y = intScheme->doNeedEy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->D1y) : NULLPTR;
        intScheme->D1z = intScheme->doNeedEz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->D1z) : NULLPTR;
        intScheme->B1x = intScheme->doNeedHx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->B1x) : NULLPTR;
        intScheme->B1y = intScheme->doNeedHy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->B1y) : NULLPTR;
        intScheme->B1z = intScheme->doNeedHz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->B1z) : NULLPTR;
      }

      intScheme->SigmaX = intScheme->doNeedSigmaX ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->SigmaX) : NULLPTR;
      intScheme->SigmaY = intScheme->doNeedSigmaY ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->SigmaY) : NULLPTR;
      intScheme->SigmaZ = intScheme->doNeedSigmaZ ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->SigmaZ) : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      intScheme->ExAmplitude = intScheme->doNeedEx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->ExAmplitude) : NULLPTR;
      intScheme->EyAmplitude = intScheme->doNeedEy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->EyAmplitude) : NULLPTR;
      intScheme->EzAmplitude = intScheme->doNeedEz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->EzAmplitude) : NULLPTR;
      intScheme->HxAmplitude = intScheme->doNeedHx ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->HxAmplitude) : NULLPTR;
      intScheme->HyAmplitude = intScheme->doNeedHy ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->HyAmplitude) : NULLPTR;
      intScheme->HzAmplitude = intScheme->doNeedHz ? new TGrid1<TC> (blockSize, bufSize, cpuScheme->HzAmplitude) : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      intScheme->OmegaPE = new TGrid1<TC> (blockSize, bufSize, cpuScheme->OmegaPE);
      intScheme->GammaE = new TGrid1<TC> (blockSize, bufSize, cpuScheme->GammaE);
      intScheme->OmegaPM = new TGrid1<TC> (blockSize, bufSize, cpuScheme->OmegaPM);
      intScheme->GammaM = new TGrid1<TC> (blockSize, bufSize, cpuScheme->GammaM);
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      TC one (1, 1, 1
#ifdef DEBUG_INFO
              , intScheme->ct1, intScheme->ct2, intScheme->ct3
#endif
              );

      intScheme->EInc = new TGrid1<GridCoordinate1D> (GridCoordinate1D (500*(cpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                                , CoordinateType::X
#endif
                                                               ), one, cpuScheme->EInc);
      intScheme->HInc = new TGrid1<GridCoordinate1D> (GridCoordinate1D (500*(cpuScheme->yeeLayout->getSize ().get1 ())
#ifdef DEBUG_INFO
                                                                , CoordinateType::X
#endif
                                                               ), one, cpuScheme->HInc);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
  CUDA_HOST
  static void allocateGridsOnGPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *gpuScheme);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
  CUDA_HOST
  static void
  copyGridsFromCPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *gpuScheme,
                    TCoord<grid_coord, true> start,
                    TCoord<grid_coord, true> end)
  {
    gpuScheme->Eps->copyFromCPU (start, end);
    gpuScheme->Mu->copyFromCPU (start, end);

    gpuScheme->Ex->copyFromCPU (start, end);
    gpuScheme->Ey->copyFromCPU (start, end);
    gpuScheme->Ez->copyFromCPU (start, end);
    gpuScheme->Hx->copyFromCPU (start, end);
    gpuScheme->Hy->copyFromCPU (start, end);
    gpuScheme->Hz->copyFromCPU (start, end);

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      gpuScheme->Dx->copyFromCPU (start, end);
      gpuScheme->Dy->copyFromCPU (start, end);
      gpuScheme->Dz->copyFromCPU (start, end);
      gpuScheme->Bx->copyFromCPU (start, end);
      gpuScheme->By->copyFromCPU (start, end);
      gpuScheme->Bz->copyFromCPU (start, end);

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuScheme->D1x->copyFromCPU (start, end);
        gpuScheme->D1y->copyFromCPU (start, end);
        gpuScheme->D1z->copyFromCPU (start, end);
        gpuScheme->B1x->copyFromCPU (start, end);
        gpuScheme->B1y->copyFromCPU (start, end);
        gpuScheme->B1z->copyFromCPU (start, end);
      }

      gpuScheme->SigmaX->copyFromCPU (start, end);
      gpuScheme->SigmaY->copyFromCPU (start, end);
      gpuScheme->SigmaZ->copyFromCPU (start, end);
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      gpuScheme->ExAmplitude->copyFromCPU (start, end);
      gpuScheme->EyAmplitude->copyFromCPU (start, end);
      gpuScheme->EzAmplitude->copyFromCPU (start, end);
      gpuScheme->HxAmplitude->copyFromCPU (start, end);
      gpuScheme->HyAmplitude->copyFromCPU (start, end);
      gpuScheme->HzAmplitude->copyFromCPU (start, end);
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
      TCoord<grid_coord, true> zero (0, 0, 0
#ifdef DEBUG_INFO
              , gpuScheme->ct1, gpuScheme->ct2, gpuScheme->ct3
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

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
  CUDA_HOST
  static void copyGridsToGPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *intScheme,
                              InternalSchemeBase<Type, TCoord, layout_type, TGrid> *gpuScheme);

  static FieldValue approximateIncidentWaveHelper (FPValue d, Grid<GridCoordinate1D> *FieldInc)
  {
    FPValue coordD1 = (FPValue) ((grid_coord) d);
    FPValue coordD2 = coordD1 + 1;
    FPValue proportionD2 = d - coordD1;
    FPValue proportionD1 = 1 - proportionD2;

    GridCoordinate1D pos1 ((grid_coord) coordD1
#ifdef DEBUG_INFO
                              , FieldInc->getSize ().getType1 ()
#endif
                          );
    GridCoordinate1D pos2 ((grid_coord) coordD2
#ifdef DEBUG_INFO
                              , FieldInc->getSize ().getType1 ()
#endif
                          );

    FieldPointValue *val1 = FieldInc->getFieldPointValue (pos1);
    FieldPointValue *val2 = FieldInc->getFieldPointValue (pos2);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    return val1->getPrevValue () * proportionD1 + val2->getPrevValue () * proportionD2;
#else
    ALWAYS_ASSERT (0);
#endif
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWave (TCoord<FPValue, true>, TCoord<FPValue, true>, FPValue, Grid<GridCoordinate1D> *, FPValue, FPValue);

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, Grid<GridCoordinate1D> *EInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.0, EInc, incAngle1, incAngle2);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, Grid<GridCoordinate1D> *HInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.5, HInc, incAngle1, incAngle2);
  }

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
  template <SchemeType_t Type, LayoutType layout_type>
  static
  void allocateParallelGrids (ParallelYeeGridLayout<Type, layout_type> *pLayout,
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
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */
};

#endif /* !INTERNAL_SCHEME_HELPER_H */

#ifndef INTERNAL_SCHEME_HELPER_H
#define INTERNAL_SCHEME_HELPER_H

#include "GridInterface.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"

class InternalSchemeHelper
{
public:

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

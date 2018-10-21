#ifndef INTERNAL_SCHEME_H
#define INTERNAL_SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"
#include "InternalSchemeHelper.h"

/**
 * Dependencies between Schemes ('|' and '<-' are used for inheritance, '!' is used for usage):
 *
 *                          Scheme1D_stype,               Scheme2D_stype,               Scheme3D_stype
 *                             |        !                    |        !                    |        !
 *                             |        !                    |        !                    |        !
 * ShemeBase           <--  Scheme1D,   !                 Scheme2D,   !                 Scheme3D    !
 *                                      !                             !                             !
 *                                      !                             !                             !
 *                          InternalScheme1D_stype_gtype, InternalScheme2D_stype_gtype, InternalScheme3D_stype_gtype
 *                             |                             |                             |
 *                             |                             |                             |
 *                          InternalScheme1D_stype,       InternalScheme2D_stype,       InternalScheme3D_stype
 *                             |                             |                             |
 *                             |                             |                             |
 * InternalSchemeBase  <--  InternalScheme1D,             InternalScheme2D,             InternalScheme3D
 */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
class InternalSchemeBase
{
  friend class InternalSchemeHelper;

protected:

  /**
   * Different types of template coordinates
   */
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  /**
   * Flag whether scheme is initialized
   */
  bool isInitialized;

  /**
   * Yee grid layout, which is being used for computations
   */
  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout;

  /**
   * Coordinate types (some might be CoordinateType::NONE)
   */
  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;

  /**
   * Field grids
   */
  TGrid<TC> *Ex;
  TGrid<TC> *Ey;
  TGrid<TC> *Ez;
  TGrid<TC> *Hx;
  TGrid<TC> *Hy;
  TGrid<TC> *Hz;

  TGrid<TC> *Dx;
  TGrid<TC> *Dy;
  TGrid<TC> *Dz;
  TGrid<TC> *Bx;
  TGrid<TC> *By;
  TGrid<TC> *Bz;

  /**
   * Auxiliary field grids
   */
  TGrid<TC> *D1x;
  TGrid<TC> *D1y;
  TGrid<TC> *D1z;
  TGrid<TC> *B1x;
  TGrid<TC> *B1y;
  TGrid<TC> *B1z;

  /**
   * Amplitude field grids
   */
  TGrid<TC> *ExAmplitude;
  TGrid<TC> *EyAmplitude;
  TGrid<TC> *EzAmplitude;
  TGrid<TC> *HxAmplitude;
  TGrid<TC> *HyAmplitude;
  TGrid<TC> *HzAmplitude;

  /**
   * Material grids
   */
  TGrid<TC> *Eps;
  TGrid<TC> *Mu;

  /**
   * Sigmas
   */
  TGrid<TC> *SigmaX;
  TGrid<TC> *SigmaY;
  TGrid<TC> *SigmaZ;

  /**
   * Metamaterial grids
   */
  TGrid<TC> *OmegaPE;
  TGrid<TC> *GammaE;

  TGrid<TC> *OmegaPM;
  TGrid<TC> *GammaM;

  /**
   * Auxiliary TF/SF 1D grids
   */
  TGrid<GridCoordinate1D> *EInc;
  TGrid<GridCoordinate1D> *HInc;

  /**
   * Wave length analytical
   */
  FPValue sourceWaveLength;

  /**
   * Wave length numerical
   */
  FPValue sourceWaveLengthNumerical;

  /**
   * Wave frequency
   */
  FPValue sourceFrequency;

  /**
   * Wave relative phase velocity
   */
  FPValue relPhaseVelocity;

  /**
   * Courant number
   */
  FPValue courantNum;

  /**
   * dx (step in space)
   */
  FPValue gridStep;

  /**
   * dt (step in time)
   */
  FPValue gridTimeStep;

  TC leftNTFF;
  TC rightNTFF;

  bool useParallel;

  SourceCallBack ExBorder;
  SourceCallBack ExInitial;

  SourceCallBack EyBorder;
  SourceCallBack EyInitial;

  SourceCallBack EzBorder;
  SourceCallBack EzInitial;

  SourceCallBack HxBorder;
  SourceCallBack HxInitial;

  SourceCallBack HyBorder;
  SourceCallBack HyInitial;

  SourceCallBack HzBorder;
  SourceCallBack HzInitial;

  SourceCallBack Jx;
  SourceCallBack Jy;
  SourceCallBack Jz;
  SourceCallBack Mx;
  SourceCallBack My;
  SourceCallBack Mz;

  SourceCallBack ExExact;
  SourceCallBack EyExact;
  SourceCallBack EzExact;
  SourceCallBack HxExact;
  SourceCallBack HyExact;
  SourceCallBack HzExact;

  /*
   * TODO: maybe add separate for Dx, etc.
   */
  const bool doNeedEx;
  const bool doNeedEy;
  const bool doNeedEz;
  const bool doNeedHx;
  const bool doNeedHy;
  const bool doNeedHz;

  const bool doNeedSigmaX;
  const bool doNeedSigmaY;
  const bool doNeedSigmaZ;

protected:

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
  CUDA_HOST
  virtual void allocateParallelGrids () { ALWAYS_ASSERT (0); }
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

  CUDA_HOST
  virtual void allocateGrids () { ALWAYS_ASSERT (0); }
  CUDA_HOST
  virtual void allocateGridsInc () { ALWAYS_ASSERT (0); }

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<Type, TCoord, layout_type, Grid> *cpuScheme, TC blockSize, TC bufSize) { ALWAYS_ASSERT (0); }
  CUDA_HOST
  virtual void allocateGridsOnGPU () { ALWAYS_ASSERT (0); }

  CUDA_HOST
  virtual void copyGridsFromCPU (TC start, TC end) { ALWAYS_ASSERT (0); }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<Type, TCoord, layout_type, CudaGrid> *gpuScheme) { ALWAYS_ASSERT (0); }

  CUDA_HOST
  virtual void initCoordTypes () { ALWAYS_ASSERT (0); }

  CUDA_DEVICE CUDA_HOST
  virtual bool doSkipBorderFunc (TC, TGrid<TC> *) { ALWAYS_ASSERT (0); return false; }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
#endif /* ENABLE_ASSERTS */

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &, TC, TC, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStep (time_step, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepInit (TGrid<TC> **, GridType *, TGrid<TC> **, GridType *, TGrid<TC> **, GridType *, TGrid<TC> **, GridType *,
    TGrid<TC> **, GridType *, TGrid<TC> **, GridType *, TGrid<TC> **, GridType *, TGrid<TC> **, TGrid<TC> **,
    TGrid<TC> **, GridType *, TGrid<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *);

  template <uint8_t grid_type, bool usePML>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIteration (time_step, TC, TGrid<TC> *, GridType, TGrid<TC> *, GridType, TGrid<TC> *, TGrid<TC> *, SourceCallBack, FPValue);

  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, TGrid<TC> *, TGrid<TC> *, GridType,
       TGrid<TC> *, GridType,  TGrid<TC> *, GridType,  TGrid<TC> *, GridType, FPValue);

  template <bool useMetamaterials>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationPML (time_step, TC, TGrid<TC> *, TGrid<TC> *, TGrid<TC> *, GridType, GridType,
       TGrid<TC> *, GridType,  TGrid<TC> *, GridType,  TGrid<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationBorder (time_step, TC, TGrid<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationExact (time_step, TC, TGrid<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);

  template<uint8_t EnumVal>
  CUDA_DEVICE CUDA_HOST
  void performPointSourceCalc (time_step);

  CUDA_DEVICE CUDA_HOST
  FieldValue calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                        FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                        FPValue Ca, FPValue Cb, FPValue delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return prev * Ca + tmp * Cb;
  }

  CUDA_DEVICE CUDA_HOST
  FieldValue calcFieldDrude (FieldValue curDOrB, FieldValue prevDOrB, FieldValue prevPrevDOrB,
                             FieldValue prevEOrH, FieldValue prevPrevEOrH,
                             FPValue b0, FPValue b1, FPValue b2, FPValue a1, FPValue a2)
  {
    return curDOrB * b0 + prevDOrB * b1 + prevPrevDOrB * b2 - prevEOrH * a1 - prevPrevEOrH * a2;
  }

  CUDA_DEVICE CUDA_HOST
  FieldValue calcFieldFromDOrB (FieldValue prevEOrH, FieldValue curDOrB, FieldValue prevDOrB,
                                FPValue Ca, FPValue Cb, FPValue Cc)
  {
    return prevEOrH * Ca + curDOrB * Cb - prevDOrB * Cc;
  }

public:

  CUDA_HOST
  InternalSchemeBase ();

  CUDA_HOST
  virtual ~InternalSchemeBase ();

  CUDA_HOST
  void
  init (YeeGridLayout<Type, TCoord, layout_type> *layout, bool parallelLayout);

  CUDA_HOST
  void
  initFromCPU (InternalSchemeBase<Type, TCoord, layout_type, Grid> *cpuScheme, TC, TC);

  CUDA_HOST
  void
  initOnGPU ();

  CUDA_HOST
  void
  copyFromCPU (TCoord<grid_coord, true>, TCoord<grid_coord, true>);

  CUDA_HOST
  void
  copyToGPU (InternalSchemeBase<Type, TCoord, layout_type, CudaGrid> *gpuScheme);

  CUDA_HOST
  void
  initScheme (FPValue, FPValue);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void performFieldSteps (time_step t, TC Start, TC End)
  {
    /*
     * TODO: remove check performed on each iteration
     */
    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, true, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, true, false> (t, Start, End);
      }
    }
    else
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, false, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, false, false> (t, Start, End);
      }
    }

    bool doUsePointSource;
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEz ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHz ();
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doUsePointSource)
    {
      performPointSourceCalc<grid_type> (t);
    }
  }

  CUDA_DEVICE CUDA_HOST
  void performPlaneWaveESteps (time_step);
  CUDA_DEVICE CUDA_HOST
  void performPlaneWaveHSteps (time_step);

  CUDA_DEVICE CUDA_HOST
  FieldValue approximateIncidentWaveE (TCFP pos);
  CUDA_DEVICE CUDA_HOST
  FieldValue approximateIncidentWaveH (TCFP pos);

  bool getDoNeedEx () const { return doNeedEx; }
  bool getDoNeedEy () const { return doNeedEy; }
  bool getDoNeedEz () const { return doNeedEz; }

  bool getDoNeedHx () const { return doNeedHx; }
  bool getDoNeedHy () const { return doNeedHy; }
  bool getDoNeedHz () const { return doNeedHz; }

  TGrid<TC> * getEx ()
  {
    ASSERT (Ex);
    ASSERT (doNeedEx);
    return Ex;
  }
  TGrid<TC> * getEy ()
  {
    ASSERT (Ey);
    ASSERT (doNeedEy);
    return Ey;
  }
  TGrid<TC> * getEz ()
  {
    ASSERT (Ez);
    ASSERT (doNeedEz);
    return Ez;
  }

  TGrid<TC> * getHx ()
  {
    ASSERT (Hx);
    ASSERT (doNeedHx);
    return Hx;
  }
  TGrid<TC> * getHy ()
  {
    ASSERT (Hy);
    ASSERT (doNeedHy);
    return Hy;
  }
  TGrid<TC> * getHz ()
  {
    ASSERT (Hz);
    ASSERT (doNeedHz);
    return Hz;
  }

  TGrid<TC> * getEps ()
  {
    ASSERT (Eps);
    return Eps;
  }
  TGrid<TC> * getMu ()
  {
    ASSERT (Mu);
    return Mu;
  }

  TGrid<TC> * getSigmaX ()
  {
    ASSERT (SigmaX);
    ASSERT (doNeedSigmaX);
    return SigmaX;
  }
  TGrid<TC> * getSigmaY ()
  {
    ASSERT (SigmaY);
    ASSERT (doNeedSigmaY);
    return SigmaY;
  }
  TGrid<TC> * getSigmaZ ()
  {
    ASSERT (SigmaZ);
    ASSERT (doNeedSigmaZ);
    return SigmaZ;
  }

  TGrid<TC> * getOmegaPE ()
  {
    ASSERT (OmegaPE);
    return OmegaPE;
  }
  TGrid<TC> * getOmegaPM ()
  {
    ASSERT (OmegaPM);
    return OmegaPM;
  }

  TGrid<TC> * getGammaE ()
  {
    ASSERT (GammaE);
    return GammaE;
  }
  TGrid<TC> * getGammaM ()
  {
    ASSERT (GammaM);
    return GammaM;
  }
};

/*
 * Dimension specific
 */

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D: public InternalSchemeBase<Type, GridCoordinate1DTemplate, layout_type, TGrid>
{
protected:

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
  CUDA_HOST
  virtual void allocateParallelGrids () CXX11_OVERRIDE_FINAL;
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

  CUDA_DEVICE CUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate1D, TGrid<GridCoordinate1D> *) CXX11_OVERRIDE_FINAL;

};

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D: public InternalSchemeBase<Type, GridCoordinate2DTemplate, layout_type, TGrid>
{
protected:

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
  CUDA_HOST
  virtual void allocateParallelGrids () CXX11_OVERRIDE_FINAL;
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

  CUDA_DEVICE CUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate2D, TGrid<GridCoordinate2D> *) CXX11_OVERRIDE_FINAL;

};

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
class InternalScheme3D: public InternalSchemeBase<Type, GridCoordinate3DTemplate, layout_type, TGrid>
{
protected:

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
  CUDA_HOST
  virtual void allocateParallelGrids () CXX11_OVERRIDE_FINAL;
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

  CUDA_DEVICE CUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate3D, TGrid<GridCoordinate3D> *) CXX11_OVERRIDE_FINAL;

};

/*
 * Scheme type specific
 */

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_ExHy: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_ExHz: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_EyHx: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_EyHz: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_EzHx: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme1D_EzHy: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct2 = CoordinateType::NONE;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TEx: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TEy: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TEz: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TMx: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TMy: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Z;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme2D_TMz: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, TGrid>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type, template <typename> class TGrid>
class InternalScheme3D_3D: public InternalScheme3D<(static_cast<SchemeType_t> (SchemeType::Dim3)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, TGrid>::ct1 = CoordinateType::X;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, TGrid>::ct2 = CoordinateType::Y;
    InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, TGrid>::ct3 = CoordinateType::Z;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
#endif /* ENABLE_ASSERTS */
};

/*
 * Grid specific
 */

template <LayoutType layout_type>
class InternalScheme1D_ExHy_Grid: public InternalScheme1D_ExHy<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_ExHz_Grid: public InternalScheme1D_ExHz<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EyHx_Grid: public InternalScheme1D_EyHx<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EyHz_Grid: public InternalScheme1D_EyHz<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EzHx_Grid: public InternalScheme1D_EzHx<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EzHy_Grid: public InternalScheme1D_EzHy<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEx_Grid: public InternalScheme2D_TEx<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEy_Grid: public InternalScheme2D_TEy<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEz_Grid: public InternalScheme2D_TEz<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMx_Grid: public InternalScheme2D_TMx<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMy_Grid: public InternalScheme2D_TMy<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMz_Grid: public InternalScheme2D_TMz<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid> (this, layout);
  }
};

template <LayoutType layout_type>
class InternalScheme3D_3D_Grid: public InternalScheme3D_3D<layout_type, Grid>
{
protected:

  CUDA_HOST
  virtual void allocateGrids () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGrids<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid> (this, layout);
  }
  CUDA_HOST
  virtual void allocateGridsInc () CXX11_OVERRIDE_FINAL
  {
    YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *layout = InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid> (this, layout);
  }
};

#ifdef CUDA_ENABLED

template <LayoutType layout_type>
class InternalScheme1D_ExHy_CudaGrid: public InternalScheme1D_ExHy<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_ExHz_CudaGrid: public InternalScheme1D_ExHz<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EyHx_CudaGrid: public InternalScheme1D_EyHx<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EyHz_CudaGrid: public InternalScheme1D_EyHz<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EzHx_CudaGrid: public InternalScheme1D_EzHx<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme1D_EzHy_CudaGrid: public InternalScheme1D_EzHy<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate1D blockSize, GridCoordinate1D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate1D start, GridCoordinate1D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEx_CudaGrid: public InternalScheme2D_TEx<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEy_CudaGrid: public InternalScheme2D_TEy<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TEz_CudaGrid: public InternalScheme2D_TEz<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMx_CudaGrid: public InternalScheme2D_TMx<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMy_CudaGrid: public InternalScheme2D_TMy<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme2D_TMz_CudaGrid: public InternalScheme2D_TMz<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate2D blockSize, GridCoordinate2D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate2D start, GridCoordinate2D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

template <LayoutType layout_type>
class InternalScheme3D_3D_CudaGrid: public InternalScheme3D_3D<layout_type, CudaGrid>
{
protected:

  CUDA_HOST
  virtual void allocateGridsFromCPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, Grid> *cpuScheme, GridCoordinate3D blockSize, GridCoordinate3D bufSize) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, CudaGrid> (this, cpuScheme, blockSize, bufSize);
  }
  CUDA_HOST
  virtual void allocateGridsOnGPU () CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::allocateGridsOnGPU<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, CudaGrid> (this);
  }

  CUDA_HOST
  virtual void copyGridsFromCPU (GridCoordinate3D start, GridCoordinate3D end) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsFromCPU<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, CudaGrid> (this, start, end);
  }
  CUDA_HOST
  virtual void copyGridsToGPU (InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, CudaGrid> *gpuScheme) CXX11_OVERRIDE_FINAL
  {
    InternalSchemeHelper::copyGridsToGPU<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type, CudaGrid> (this, gpuScheme);
  }
};

#endif /* CUDA_ENABLED */

#include "InternalScheme.template.h"

#endif /* !INTERNAL_SCHEME_H */

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
};

#endif /* CUDA_ENABLED */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template <uint8_t grid_type>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateTFSF (TC posAbs,
                                                       FieldValue &valOpposite11,
                                                       FieldValue &valOpposite12,
                                                       FieldValue &valOpposite21,
                                                       FieldValue &valOpposite22,
                                                       TC pos11,
                                                       TC pos12,
                                                       TC pos21,
                                                       TC pos22)
{
  bool doNeedUpdate11;
  bool doNeedUpdate12;
  bool doNeedUpdate21;
  bool doNeedUpdate22;

  bool isRevertVals;

  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFExAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEx);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFEyAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEy);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::FRONT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFEzAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEz);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHxAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHx);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::FRONT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP);

      isRevertVals = false;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHyAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHy);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::FRONT);

      isRevertVals = false;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHzAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHz);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT);

      isRevertVals = false;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  TC auxPos1;
  TC auxPos2;
  FieldValue diff1;
  FieldValue diff2;

  if (isRevertVals)
  {
    if (doNeedUpdate11)
    {
      auxPos1 = pos12;
    }
    else if (doNeedUpdate12)
    {
      auxPos1 = pos11;
    }

    if (doNeedUpdate21)
    {
      auxPos2 = pos22;
    }
    else if (doNeedUpdate22)
    {
      auxPos2 = pos21;
    }
  }
  else
  {
    if (doNeedUpdate11)
    {
      auxPos1 = pos11;
    }
    else if (doNeedUpdate12)
    {
      auxPos1 = pos12;
    }

    if (doNeedUpdate21)
    {
      auxPos2 = pos21;
    }
    else if (doNeedUpdate22)
    {
      auxPos2 = pos22;
    }
  }

  if (doNeedUpdate11 || doNeedUpdate12)
  {
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        if (doNeedHz)
        {
          TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        if (doNeedHx)
        {
          TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        if (doNeedHy)
        {
          TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        if (doNeedEy)
        {
          TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        if (doNeedEz)
        {
          TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        if (doNeedEx)
        {
          TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  if (doNeedUpdate21 || doNeedUpdate22)
  {
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        if (doNeedHy)
        {
          TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        if (doNeedHz)
        {
          TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        if (doNeedHx)
        {
          TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        if (doNeedEz)
        {
          TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        if (doNeedEx)
        {
          TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        if (doNeedEy)
        {
          TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord)) * FPValue (-1.0);
        }

        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  if (isRevertVals)
  {
    if (doNeedUpdate11)
    {
      valOpposite12 -= diff1;
    }
    else if (doNeedUpdate12)
    {
      valOpposite11 -= diff1;
    }

    if (doNeedUpdate21)
    {
      valOpposite22 -= diff2;
    }
    else if (doNeedUpdate22)
    {
      valOpposite21 -= diff2;
    }
  }
  else
  {
    if (doNeedUpdate11)
    {
      valOpposite11 -= diff1;
    }
    else if (doNeedUpdate12)
    {
      valOpposite12 -= diff1;
    }

    if (doNeedUpdate21)
    {
      valOpposite21 -= diff2;
    }
    else if (doNeedUpdate22)
    {
      valOpposite22 -= diff2;
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStep (time_step t, TC start, TC end)
{
  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

  FPValue eps0 = PhysicsConst::Eps0;

  Grid<TC> *grid = NULLPTR;
  GridType gridType = GridType::NONE;

  Grid<TC> *materialGrid = NULLPTR;
  GridType materialGridType = GridType::NONE;

  Grid<TC> *materialGrid1 = NULLPTR;
  GridType materialGridType1 = GridType::NONE;

  Grid<TC> *materialGrid2 = NULLPTR;
  GridType materialGridType2 = GridType::NONE;

  Grid<TC> *materialGrid3 = NULLPTR;
  GridType materialGridType3 = GridType::NONE;

  Grid<TC> *materialGrid4 = NULLPTR;
  GridType materialGridType4 = GridType::NONE;

  Grid<TC> *materialGrid5 = NULLPTR;
  GridType materialGridType5 = GridType::NONE;

  Grid<TC> *oppositeGrid1 = NULLPTR;
  Grid<TC> *oppositeGrid2 = NULLPTR;

  Grid<TC> *gridPML1 = NULLPTR;
  GridType gridPMLType1 = GridType::NONE;

  Grid<TC> *gridPML2 = NULLPTR;
  GridType gridPMLType2 = GridType::NONE;

  SourceCallBack rightSideFunc = NULLPTR;
  SourceCallBack borderFunc = NULLPTR;
  SourceCallBack exactFunc = NULLPTR;

  /*
   * TODO: remove this, multiply on this at initialization
   */
  FPValue materialModifier;

  calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&grid, &gridType,
    &materialGrid, &materialGridType, &materialGrid1, &materialGridType1, &materialGrid2, &materialGridType2,
    &materialGrid3, &materialGridType3, &materialGrid4, &materialGridType4, &materialGrid5, &materialGridType5,
    &oppositeGrid1, &oppositeGrid2, &gridPML1, &gridPMLType1, &gridPML2, &gridPMLType2,
    &rightSideFunc, &borderFunc, &exactFunc, &materialModifier);

  GridCoordinate3D start3D;
  GridCoordinate3D end3D;

  expandTo3DStartEnd (start, end, start3D, end3D, ct1, ct2, ct3);

  // TODO: remove this check for each iteration
  if (t > 0)
  {
    for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
      {
        // TODO: check that this is optimized out in case 2D mode
        for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIteration<grid_type, usePML> (t, pos, grid, gridType, materialGrid, materialGridType,
                                                          oppositeGrid1, oppositeGrid2, rightSideFunc, materialModifier);
        }
      }
    }

    if (usePML)
    {
      if (useMetamaterials)
      {
#ifdef TWO_TIME_STEPS
        for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
          {
            // TODO: check that this loop is optimized out
            for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
            {
              TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
              calculateFieldStepIterationPMLMetamaterials (t, pos, grid, gridPML1, gridType,
                materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
                materialModifier);
            }
          }
        }
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
      }

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            calculateFieldStepIterationPML<useMetamaterials> (t, pos, grid, gridPML1, gridPML2, gridType, gridPMLType1,
              materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
              materialModifier);
          }
        }
      }
    }
  }

  if (borderFunc != NULLPTR)
  {
    GridCoordinate3D startBorder;
    GridCoordinate3D endBorder;

    expandTo3DStartEnd (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3),
                        grid->getSize (),
                        startBorder,
                        endBorder,
                        ct1, ct2, ct3);

    for (grid_coord i = startBorder.get1 (); i < endBorder.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startBorder.get2 (); j < endBorder.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startBorder.get3 (); k < endBorder.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIterationBorder<grid_type> (t, pos, grid, borderFunc);
        }
      }
    }
  }

  if (exactFunc != NULLPTR)
  {
    FPValue normRe = 0.0;
    FPValue normIm = 0.0;
    FPValue normMod = 0.0;

    FPValue maxRe = 0.0;
    FPValue maxIm = 0.0;
    FPValue maxMod = 0.0;

    GridCoordinate3D startNorm = start3D;
    GridCoordinate3D endNorm = end3D;

    if (SOLVER_SETTINGS.getExactSolutionCompareStartX () != 0)
    {
      startNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareStartX ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareStartY () != 0)
    {
      startNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareStartY ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareStartZ () != 0)
    {
      startNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareStartZ ());
    }

    if (SOLVER_SETTINGS.getExactSolutionCompareEndX () != 0)
    {
      endNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareEndX ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareEndY () != 0)
    {
      endNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareEndY ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareEndZ () != 0)
    {
      endNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareEndZ ());
    }

    Grid<TC> *normGrid = grid;
    if (usePML)
    {
      grid = gridPML2;
    }

    for (grid_coord i = startNorm.get1 (); i < endNorm.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startNorm.get2 (); j < endNorm.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startNorm.get3 (); k < endNorm.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIterationExact<grid_type> (t, pos, grid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
        }
      }
    }

#ifdef COMPLEX_FIELD_VALUES
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
    normIm = sqrt (normIm / grid->getSize ().calculateTotalCoord ());
    normMod = sqrt (normMod / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
      grid->getName (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
      grid->getName (), t, normRe, normRe * 100.0 / maxRe);
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepInit (TGrid<TC> **grid, GridType *gridType, TGrid<TC> **materialGrid, GridType *materialGridType, TGrid<TC> **materialGrid1, GridType *materialGridType1,
TGrid<TC> **materialGrid2, GridType *materialGridType2, TGrid<TC> **materialGrid3, GridType *materialGridType3, TGrid<TC> **materialGrid4, GridType *materialGridType4,
TGrid<TC> **materialGrid5, GridType *materialGridType5, TGrid<TC> **oppositeGrid1, TGrid<TC> **oppositeGrid2, TGrid<TC> **gridPML1, GridType *gridPMLType1, TGrid<TC> **gridPML2, GridType *gridPMLType2,
SourceCallBack *rightSideFunc, SourceCallBack *borderFunc, SourceCallBack *exactFunc, FPValue *materialModifier)
{
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      ASSERT (doNeedEx);
      *grid = Ex;
      *gridType = GridType::EX;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hz;
      *oppositeGrid2 = Hy;

      *rightSideFunc = Jx;
      *borderFunc = ExBorder;
      *exactFunc = ExExact;

      if (usePML)
      {
        *grid = Dx;
        *gridType = GridType::DX;

        *gridPML1 = D1x;
        *gridPMLType1 = GridType::DX;

        *gridPML2 = Ex;
        *gridPMLType2 = GridType::EX;

        *materialGrid = SigmaY;
        *materialGridType = GridType::SIGMAY;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaX;
        *materialGridType4 = GridType::SIGMAX;

        *materialGrid5 = SigmaZ;
        *materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      ASSERT (doNeedEy);
      *grid = Ey;
      *gridType = GridType::EY;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hx;
      *oppositeGrid2 = Hz;

      *rightSideFunc = Jy;
      *borderFunc = EyBorder;
      *exactFunc = EyExact;

      if (usePML)
      {
        *grid = Dy;
        *gridType = GridType::DY;

        *gridPML1 = D1y;
        *gridPMLType1 = GridType::DY;

        *gridPML2 = Ey;
        *gridPMLType2 = GridType::EY;

        *materialGrid = SigmaZ;
        *materialGridType = GridType::SIGMAZ;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaY;
        *materialGridType4 = GridType::SIGMAY;

        *materialGrid5 = SigmaX;
        *materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      ASSERT (doNeedEz);
      *grid = Ez;
      *gridType = GridType::EZ;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hy;
      *oppositeGrid2 = Hx;

      *rightSideFunc = Jz;
      *borderFunc = EzBorder;
      *exactFunc = EzExact;

      if (usePML)
      {
        *grid = Dz;
        *gridType = GridType::DZ;

        *gridPML1 = D1z;
        *gridPMLType1 = GridType::DZ;

        *gridPML2 = Ez;
        *gridPMLType2 = GridType::EZ;

        *materialGrid = SigmaX;
        *materialGridType = GridType::SIGMAX;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaZ;
        *materialGridType4 = GridType::SIGMAZ;

        *materialGrid5 = SigmaY;
        *materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      ASSERT (doNeedHx);
      *grid = Hx;
      *gridType = GridType::HX;

      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ey;
      *oppositeGrid2 = Ez;

      *rightSideFunc = Mx;
      *borderFunc = HxBorder;
      *exactFunc = HxExact;

      if (usePML)
      {
        *grid = Bx;
        *gridType = GridType::BX;

        *gridPML1 = B1x;
        *gridPMLType1 = GridType::BX;

        *gridPML2 = Hx;
        *gridPMLType2 = GridType::HX;

        *materialGrid = SigmaY;
        *materialGridType = GridType::SIGMAY;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaX;
        *materialGridType4 = GridType::SIGMAX;

        *materialGrid5 = SigmaZ;
        *materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      ASSERT (doNeedHy);
      *grid = Hy;
      *gridType = GridType::HY;

      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ez;
      *oppositeGrid2 = Ex;

      *rightSideFunc = My;
      *borderFunc = HyBorder;
      *exactFunc = HyExact;

      if (usePML)
      {
        *grid = By;
        *gridType = GridType::BY;

        *gridPML1 = B1y;
        *gridPMLType1 = GridType::BY;

        *gridPML2 = Hy;
        *gridPMLType2 = GridType::HY;

        *materialGrid = SigmaZ;
        *materialGridType = GridType::SIGMAZ;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaY;
        *materialGridType4 = GridType::SIGMAY;

        *materialGrid5 = SigmaX;
        *materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      ASSERT (doNeedHz);
      *grid = Hz;
      *gridType = GridType::HZ;
      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ex;
      *oppositeGrid2 = Ey;

      *rightSideFunc = Mz;
      *borderFunc = HzBorder;
      *exactFunc = HzExact;

      if (usePML)
      {
        *grid = Bz;
        *gridType = GridType::BZ;

        *gridPML1 = B1z;
        *gridPMLType1 = GridType::BZ;

        *gridPML2 = Hz;
        *gridPMLType2 = GridType::HZ;

        *materialGrid = SigmaX;
        *materialGridType = GridType::SIGMAX;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaZ;
        *materialGridType4 = GridType::SIGMAZ;

        *materialGrid5 = SigmaY;
        *materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template<uint8_t grid_type, bool usePML>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepIteration (time_step t,
                                                               TC pos,
                                                               TGrid<TC> *grid,
                                                               GridType gridType,
                                                               TGrid<TC> *materialGrid,
                                                               GridType materialGridType,
                                                               TGrid<TC> *oppositeGrid1,
                                                               TGrid<TC> *oppositeGrid2,
                                                               SourceCallBack rightSideFunc,
                                                               FPValue materialModifier)
{
  FPValue eps0 = PhysicsConst::Eps0;

  // TODO: add getTotalPositionDiff here, which will be called before loop
  TC posAbs = grid->getTotalPosition (pos);
  // TODO: [possible] move 1D gridValues to 3D gridValues array
  FieldPointValue *valField = grid->getFieldPointValue (pos);

  FPValue material = materialGrid ? yeeLayout->getMaterial (posAbs, gridType, materialGrid, materialGridType) : 0;

  TC pos11 = pos;
  TC pos12 = pos;
  TC pos21 = pos;
  TC pos22 = pos;

  TCFP coordFP;
  FPValue timestep;

  FPValue k_mod;
  FPValue Ca;
  FPValue Cb;

  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      pos11 = pos11 + yeeLayout->getExCircuitElementDiff (LayoutDirection::DOWN);
      pos12 = pos12 + yeeLayout->getExCircuitElementDiff (LayoutDirection::UP);
      pos21 = pos21 + yeeLayout->getExCircuitElementDiff (LayoutDirection::BACK);
      pos22 = pos22 + yeeLayout->getExCircuitElementDiff (LayoutDirection::FRONT);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getExCoordFP (posAbs);
        timestep = t;
      }

      FPValue k_y = 1;
      k_mod = k_y;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      pos11 = pos11 + yeeLayout->getEyCircuitElementDiff (LayoutDirection::BACK);
      pos12 = pos12 + yeeLayout->getEyCircuitElementDiff (LayoutDirection::FRONT);
      pos21 = pos21 + yeeLayout->getEyCircuitElementDiff (LayoutDirection::LEFT);
      pos22 = pos22 + yeeLayout->getEyCircuitElementDiff (LayoutDirection::RIGHT);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getEyCoordFP (posAbs);
        timestep = t;
      }

      FPValue k_z = 1;
      k_mod = k_z;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      pos11 = pos11 + yeeLayout->getEzCircuitElementDiff (LayoutDirection::LEFT);
      pos12 = pos12 + yeeLayout->getEzCircuitElementDiff (LayoutDirection::RIGHT);
      pos21 = pos21 + yeeLayout->getEzCircuitElementDiff (LayoutDirection::DOWN);
      pos22 = pos22 + yeeLayout->getEzCircuitElementDiff (LayoutDirection::UP);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getEzCoordFP (posAbs);
        timestep = t;
      }

      FPValue k_x = 1;
      k_mod = k_x;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      pos11 = pos11 + yeeLayout->getHxCircuitElementDiff (LayoutDirection::BACK);
      pos12 = pos12 + yeeLayout->getHxCircuitElementDiff (LayoutDirection::FRONT);
      pos21 = pos21 + yeeLayout->getHxCircuitElementDiff (LayoutDirection::DOWN);
      pos22 = pos22 + yeeLayout->getHxCircuitElementDiff (LayoutDirection::UP);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getHxCoordFP (posAbs);
        timestep = t + 0.5;
      }

      FPValue k_y = 1;
      k_mod = k_y;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      pos11 = pos11 + yeeLayout->getHyCircuitElementDiff (LayoutDirection::LEFT);
      pos12 = pos12 + yeeLayout->getHyCircuitElementDiff (LayoutDirection::RIGHT);
      pos21 = pos21 + yeeLayout->getHyCircuitElementDiff (LayoutDirection::BACK);
      pos22 = pos22 + yeeLayout->getHyCircuitElementDiff (LayoutDirection::FRONT);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getHyCoordFP (posAbs);
        timestep = t + 0.5;
      }

      FPValue k_z = 1;
      k_mod = k_z;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      pos11 = pos11 + yeeLayout->getHzCircuitElementDiff (LayoutDirection::DOWN);
      pos12 = pos12 + yeeLayout->getHzCircuitElementDiff (LayoutDirection::UP);
      pos21 = pos21 + yeeLayout->getHzCircuitElementDiff (LayoutDirection::LEFT);
      pos22 = pos22 + yeeLayout->getHzCircuitElementDiff (LayoutDirection::RIGHT);

      if (rightSideFunc != NULLPTR)
      {
        coordFP = yeeLayout->getHzCoordFP (posAbs);
        timestep = t + 0.5;
      }

      FPValue k_x = 1;
      k_mod = k_x;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (usePML)
  {
    Ca = (2 * eps0 * k_mod - material * gridTimeStep) / (2 * eps0 * k_mod + material * gridTimeStep);
    Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_mod + material * gridTimeStep);
  }
  else
  {
    Ca = 1.0;
    Cb = gridTimeStep / (material * materialModifier * gridStep);
  }

  // TODO: separate previous grid and current
  FieldValue prev11 = FIELDVALUE (0, 0);
  FieldValue prev12 = FIELDVALUE (0, 0);
  FieldValue prev21 = FIELDVALUE (0, 0);
  FieldValue prev22 = FIELDVALUE (0, 0);

  if (oppositeGrid1)
  {
    FieldPointValue *val11 = oppositeGrid1->getFieldPointValue (pos11);
    FieldPointValue *val12 = oppositeGrid1->getFieldPointValue (pos12);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev11 = val11->getPrevValue ();
    prev12 = val12->getPrevValue ();
#else
    ALWAYS_ASSERT (0);
#endif
  }

  if (oppositeGrid2)
  {
    FieldPointValue *val21 = oppositeGrid2->getFieldPointValue (pos21);
    FieldPointValue *val22 = oppositeGrid2->getFieldPointValue (pos22);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev21 = val21->getPrevValue ();
    prev22 = val22->getPrevValue ();
#else
    ALWAYS_ASSERT (0);
#endif
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    calculateTFSF<grid_type> (posAbs, prev11, prev12, prev21, prev22, pos11, pos12, pos21, pos22);
  }

  FieldValue prevRightSide = 0;
  if (rightSideFunc != NULLPTR)
  {
    prevRightSide = rightSideFunc (expandTo3D (coordFP * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);
  }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // TODO: precalculate Ca,Cb
  FieldValue val = calcField (valField->getPrevValue (),
                              prev12,
                              prev11,
                              prev22,
                              prev21,
                              prevRightSide,
                              Ca,
                              Cb,
                              gridStep);
#else
  ALWAYS_ASSERT (0);
#endif

  valField->setCurValue (val);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepIterationPMLMetamaterials (time_step t,
                                                                               TC pos,
                                                                               TGrid<TC> *grid,
                                                                               TGrid<TC> *gridPML,
                                                                               GridType gridType,
                                                                               TGrid<TC> *materialGrid1,
                                                                               GridType materialGridType1,
                                                                               TGrid<TC> *materialGrid2,
                                                                               GridType materialGridType2,
                                                                               TGrid<TC> *materialGrid3,
                                                                               GridType materialGridType3,
                                                                               FPValue materialModifier)
{
  TC posAbs = grid->getTotalPosition (pos);
  FieldPointValue *valField = grid->getFieldPointValue (pos);
  FieldPointValue *valField1 = gridPML->getFieldPointValue (pos);

  FPValue material1;
  FPValue material2;

  FPValue material = yeeLayout->getMetaMaterial (posAbs, gridType,
                                                 materialGrid1, materialGridType1,
                                                 materialGrid2, materialGridType2,
                                                 materialGrid3, materialGridType3,
                                                 material1, material2);

  /*
   * TODO: precalculate coefficients
   */
  FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
  FPValue a1 = (4 + 2*gridTimeStep*material2) / A;
  FPValue a2 = -8 / A;
  FPValue a3 = (4 - 2*gridTimeStep*material2) / A;
  FPValue a4 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
  FPValue a5 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;

#if defined (TWO_TIME_STEPS)
  FieldValue val = calcFieldDrude (valField->getCurValue (),
                                   valField->getPrevValue (),
                                   valField->getPrevPrevValue (),
                                   valField1->getPrevValue (),
                                   valField1->getPrevPrevValue (),
                                   a1,
                                   a2,
                                   a3,
                                   a4,
                                   a5);
  valField1->setCurValue (val);
#else
  ALWAYS_ASSERT (0);
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template <bool useMetamaterials>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepIterationPML (time_step t,
                                                                   TC pos,
                                                                   TGrid<TC> *grid,
                                                                   TGrid<TC> *gridPML1,
                                                                   TGrid<TC> *gridPML2,
                                                                   GridType gridType,
                                                                   GridType gridPMLType1,
                                                                   TGrid<TC> *materialGrid1,
                                                                   GridType materialGridType1,
                                                                   TGrid<TC> *materialGrid4,
                                                                   GridType materialGridType4,
                                                                   TGrid<TC> *materialGrid5,
                                                                   GridType materialGridType5,
                                                                   FPValue materialModifier)
{
  FPValue eps0 = PhysicsConst::Eps0;

  TC posAbs = gridPML2->getTotalPosition (pos);

  FieldPointValue *valField = gridPML2->getFieldPointValue (pos);

  FieldPointValue *valField1;

  if (useMetamaterials)
  {
    valField1 = gridPML1->getFieldPointValue (pos);
  }
  else
  {
    valField1 = grid->getFieldPointValue (pos);
  }

  FPValue material1 = materialGrid1 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1) : 0;
  FPValue material4 = materialGrid4 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4) : 0;
  FPValue material5 = materialGrid5 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5) : 0;

  FPValue modifier = material1 * materialModifier;
  if (useMetamaterials)
  {
    modifier = 1;
  }

  FPValue k_mod1 = 1;
  FPValue k_mod2 = 1;

  FPValue Ca = (2 * eps0 * k_mod2 - material5 * gridTimeStep) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
  FPValue Cb = ((2 * eps0 * k_mod1 + material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
  FPValue Cc = ((2 * eps0 * k_mod1 - material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue val = calcFieldFromDOrB (valField->getPrevValue (),
                                      valField1->getCurValue (),
                                      valField1->getPrevValue (),
                                      Ca,
                                      Cb,
                                      Cc);
#else
  ALWAYS_ASSERT (0);
#endif

  valField->setCurValue (val);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template <uint8_t grid_type>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepIterationBorder (time_step t,
                                                                      TC pos,
                                                                      TGrid<TC> *grid,
                                                                      SourceCallBack borderFunc)
{
  TC posAbs = grid->getTotalPosition (pos);

  if (doSkipBorderFunc (posAbs, grid))
  {
    return;
  }

  TCFP realCoord;
  FPValue timestep;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      realCoord = yeeLayout->getExCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      realCoord = yeeLayout->getEyCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      realCoord = yeeLayout->getEzCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      realCoord = yeeLayout->getHxCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      realCoord = yeeLayout->getHyCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      realCoord = yeeLayout->getHzCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  grid->getFieldPointValue (pos)->setCurValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep));
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template <uint8_t grid_type>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::calculateFieldStepIterationExact (time_step t,
                                                                     TC pos,
                                                                     TGrid<TC> *grid,
                                                                     SourceCallBack exactFunc,
                                                                     FPValue &normRe,
                                                                     FPValue &normIm,
                                                                     FPValue &normMod,
                                                                     FPValue &maxRe,
                                                                     FPValue &maxIm,
                                                                     FPValue &maxMod)
{
  TC posAbs = grid->getTotalPosition (pos);

  TCFP realCoord;
  FPValue timestep;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      realCoord = yeeLayout->getExCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      realCoord = yeeLayout->getEyCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      realCoord = yeeLayout->getEzCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      realCoord = yeeLayout->getHxCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      realCoord = yeeLayout->getHyCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      realCoord = yeeLayout->getHzCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  FieldValue numerical = grid->getFieldPointValue (pos)->getCurValue ();
  FieldValue exact = exactFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);

#ifdef COMPLEX_FIELD_VALUES
  FPValue modExact = sqrt (SQR (exact.real ()) + SQR (exact.imag ()));
  FPValue modNumerical = sqrt (SQR (numerical.real ()) + SQR (numerical.imag ()));

  //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact.real (), numerical.real ());

  normRe += SQR (exact.real () - numerical.real ());
  normIm += SQR (exact.imag () - numerical.imag ());
  normMod += SQR (modExact - modNumerical);

  FPValue exactAbs = fabs (exact.real ());
  if (maxRe < exactAbs)
  {
    maxRe = exactAbs;
  }

  exactAbs = fabs (exact.imag ());
  if (maxIm < exactAbs)
  {
    maxIm = exactAbs;
  }

  exactAbs = modExact;
  if (maxMod < exactAbs)
  {
    maxMod = exactAbs;
  }
#else
  normRe += SQR (exact - numerical);

  //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact, numerical);

  FPValue exactAbs = fabs (exact);
  if (maxRe < exactAbs)
  {
    maxRe = exactAbs;
  }
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
template<uint8_t EnumVal>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::performPointSourceCalc (time_step t)
{
  TGrid<TC> *grid = NULLPTR;

  switch (EnumVal)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      ASSERT (doNeedEx);
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      ASSERT (doNeedEy);
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      ASSERT (doNeedEz);
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      ASSERT (doNeedHx);
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      ASSERT (doNeedHy);
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      ASSERT (doNeedHz);
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (grid != NULLPTR);

  TC pos = TC::initAxesCoordinate (SOLVER_SETTINGS.getPointSourcePositionX (),
                                   SOLVER_SETTINGS.getPointSourcePositionY (),
                                   SOLVER_SETTINGS.getPointSourcePositionZ (),
                                   ct1, ct2, ct3);

  FieldPointValue* pointVal = grid->getFieldPointValueOrNullByAbsolutePos (pos);

  if (pointVal)
  {
#ifdef COMPLEX_FIELD_VALUES
    pointVal->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                       cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
    pointVal->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

#endif /* !INTERNAL_SCHEME_H */

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
 * ShemeBase           <--  Scheme1D,               Scheme2D,               Scheme3D
 *                             !                       !                       !
 *                             !                       !                       !
 *                          InternalScheme1D_stype, InternalScheme2D_stype, InternalScheme3D_stype
 *                             |                       |                       |
 *                             |                       |                       |
 * InternalSchemeBase  <--  InternalScheme1D,       InternalScheme2D,       InternalScheme3D
 */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalSchemeBase
{
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
  virtual void allocateParallelGrids ();
#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

  CUDA_HOST
  virtual void initCoordTypes () { ALWAYS_ASSERT (0); }

  CUDA_DEVICE CUDA_HOST
  virtual bool doSkipBorderFunc (TC, Grid<TC> *) { ALWAYS_ASSERT (0); return false; }

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
  void calculateFieldStep (time_step, TC, TC, CoordinateType, CoordinateType, CoordinateType);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepInit (Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *,
    Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, Grid<TC> **,
    Grid<TC> **, GridType *, Grid<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *);

  template <uint8_t grid_type, bool usePML>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIteration (time_step, TC, Grid<TC> *, GridType, Grid<TC> *, GridType, Grid<TC> *, Grid<TC> *, SourceCallBack, FPValue);

  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, Grid<TC> *, Grid<TC> *, GridType,
       Grid<TC> *, GridType,  Grid<TC> *, GridType,  Grid<TC> *, GridType, FPValue);

  template <bool useMetamaterials>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationPML (time_step, TC, Grid<TC> *, Grid<TC> *, Grid<TC> *, GridType, GridType,
       Grid<TC> *, GridType,  Grid<TC> *, GridType,  Grid<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationBorder (time_step, TC, Grid<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void calculateFieldStepIterationExact (time_step, TC, Grid<TC> *, SourceCallBack,
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

  // CUDA_HOST
  // init (YeeGridLayout<Type, TCoord, layout_type> *layout,
  //                                            bool parallelLayout);
  // CUDA_HOST
  // initFromCPU (InternalScheme<Type, TCoord, layout_type, Grid> *cpuScheme);

  template <uint8_t grid_type>
  CUDA_DEVICE CUDA_HOST
  void performFieldSteps (time_step t, TC Start, TC End);

  CUDA_DEVICE CUDA_HOST
  void performPlaneWaveESteps (time_step);
  CUDA_DEVICE CUDA_HOST
  void performPlaneWaveHSteps (time_step);

  CUDA_DEVICE CUDA_HOST
  FieldValue approximateIncidentWaveE (TCFP pos);
  CUDA_DEVICE CUDA_HOST
  FieldValue approximateIncidentWaveH (TCFP pos);
};

template <SchemeType_t Type, LayoutType layout_type, template <template <typename, class> > class TGrid>
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

template <SchemeType_t Type, LayoutType layout_type, template <template <typename, class> > class TGrid>
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

template <SchemeType_t Type, LayoutType layout_type, template <template <typename, class> > class TGrid>
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



template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_ExHy: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Z;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_ExHz: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Y;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_EyHx: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Z;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_EyHz: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_EzHx: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Y;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme1D_EzHy: public InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::NONE;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TEx: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Y;
    ct2 = CoordinateType::Z;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TEy: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::Z;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TEz: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::Y;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TMx: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::Y;
    ct2 = CoordinateType::Z;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TMy: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::Z;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme2D_TMz: public InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::Y;
    ct3 = CoordinateType::NONE;
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

template <LayoutType layout_type, template <template <typename, class> > class TGrid>
class InternalScheme3D_3D: public InternalScheme3D<(static_cast<SchemeType_t> (SchemeType::Dim3)), layout_type, TGrid>
{
protected:

  CUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    ct1 = CoordinateType::X;
    ct2 = CoordinateType::Y;
    ct3 = CoordinateType::Z;
  }

#ifdef ENABLE_ASSERTS
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  CUDA_DEVICE CUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
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

#include "InternalScheme.template.h"

#endif /* !INTERNAL_SCHEME_H */

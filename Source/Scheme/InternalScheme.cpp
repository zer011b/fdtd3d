#include "InternalScheme.h"

#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalScheme1D<Type, layout_type, TGrid>::allocateParallelGrids ()
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

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalScheme2D<Type, layout_type, TGrid>::allocateParallelGrids ()
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

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalScheme2D<Type, layout_type, TGrid>::allocateParallelGrids ()
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

#endif /* PARALLEL_GRID && !__CUDA_ARCH__ */

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
bool
InternalScheme1D<Type, layout_type, TGrid>::doSkipBorderFunc (GridCoordinate1D pos, TGrid<GridCoordinate1D> *grid)
{
  return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1;
}

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
bool
InternalScheme2D<Type, layout_type, TGrid>::doSkipBorderFunc (GridCoordinate2D pos, TGrid<GridCoordinate2D> *grid)
{
  return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
         && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1;
}

template <SchemeType_t Type, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
bool
InternalScheme3D<Type, layout_type, TGrid>::doSkipBorderFunc (GridCoordinate3D pos, TGrid<GridCoordinate3D> *grid)
{
  return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
         && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1
         && pos.get3 () != 0 && pos.get3 () != grid->getTotalSize ().get3 () - 1;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::InternalSchemeBase ()
  : isInitialized (false)
  , yeeLayout (NULLPTR)
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
  , sourceWaveLength (0)
  , sourceWaveLengthNumerical (0)
  , sourceFrequency (0)
  , courantNum (0)
  , gridStep (0)
  , gridTimeStep (0)
  , useParallel (false)
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
  , doNeedEx (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedEy (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedEz (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHx (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHy (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHz (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaX (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaY (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaZ (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
{
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::~InternalSchemeBase ()
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::init (YeeGridLayout<Type, TCoord, layout_type> *layout,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
FieldValue
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::approximateIncidentWaveE (TCFP pos)
{
  YeeGridLayout<Type, TCoord, layout_type> *layout = InternalSchemeBase<Type, TCoord, layout_type, TGrid>::yeeLayout;
  return InternalSchemeHelper::approximateIncidentWaveE<Type, TCoord> (pos, layout->getZeroIncCoordFP (), EInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
}
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
FieldValue
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::approximateIncidentWaveH (TCFP pos)
{
  YeeGridLayout<Type, TCoord, layout_type> *layout = InternalSchemeBase<Type, TCoord, layout_type, TGrid>::yeeLayout;
  return InternalSchemeHelper::approximateIncidentWaveH<Type, TCoord> (pos, layout->getZeroIncCoordFP (), HInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::performPlaneWaveESteps (time_step t)
{
  grid_coord size = EInc->getSize ().get1 ();

  ASSERT (size > 0);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Eps0 * gridStep);

  for (grid_coord i = 1; i < size; ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , CoordinateType::X
#endif
                          );

    FieldPointValue *valE = EInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i - 1
#ifdef DEBUG_INFO
                              , CoordinateType::X
#endif
                              );
    GridCoordinate1D posRight (i
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif
                               );

    FieldPointValue *valH1 = HInc->getFieldPointValue (posLeft);
    FieldPointValue *valH2 = HInc->getFieldPointValue (posRight);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue val = valE->getPrevValue () + (valH1->getPrevValue () - valH2->getPrevValue ()) * modifier;
#else
    ALWAYS_ASSERT (0);
#endif

    valE->setCurValue (val);
  }

  GridCoordinate1D pos (0
#ifdef DEBUG_INFO
                        , CoordinateType::X
#endif
                        );
  FieldPointValue *valE = EInc->getFieldPointValue (pos);

  FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;

#ifdef COMPLEX_FIELD_VALUES
  valE->setCurValue (FieldValue (sin (arg), cos (arg)));
#else /* COMPLEX_FIELD_VALUES */
  valE->setCurValue (sin (arg));
#endif /* !COMPLEX_FIELD_VALUES */

#ifdef ENABLE_ASSERTS
  GridCoordinate1D posEnd (size - 1, CoordinateType::X);
  ALWAYS_ASSERT (EInc->getFieldPointValue (posEnd)->getCurValue () == getFieldValueRealOnly (0.0));
#endif

  EInc->nextTimeStep ();
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_DEVICE CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::performPlaneWaveHSteps (time_step t)
{
  grid_coord size = HInc->getSize ().get1 ();

  ASSERT (size > 1);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Mu0 * gridStep);

  for (grid_coord i = 0; i < size - 1; ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , CoordinateType::X
#endif
                          );

    FieldPointValue *valH = HInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i
#ifdef DEBUG_INFO
                              , CoordinateType::X
#endif
                              );
    GridCoordinate1D posRight (i + 1
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif
                               );

    FieldPointValue *valE1 = EInc->getFieldPointValue (posLeft);
    FieldPointValue *valE2 = EInc->getFieldPointValue (posRight);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue val = valH->getPrevValue () + (valE1->getPrevValue () - valE2->getPrevValue ()) * modifier;
#else
    ALWAYS_ASSERT (0);
#endif

    valH->setCurValue (val);
  }

#ifdef ENABLE_ASSERTS
  GridCoordinate1D pos (size - 2, CoordinateType::X);
  ALWAYS_ASSERT (HInc->getFieldPointValue (pos)->getCurValue () == getFieldValueRealOnly (0.0));
#endif

  HInc->nextTimeStep ();
}

template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED, Grid>;

template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED, Grid>;
template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED, Grid>;

template class InternalSchemeBase<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED, Grid>;


template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED, Grid>;
template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED, Grid>;
template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED, Grid>;
template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED, Grid>;
template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED, Grid>;
template class InternalScheme1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED, Grid>;

template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED, Grid>;
template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED, Grid>;
template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED, Grid>;
template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED, Grid>;
template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED, Grid>;
template class InternalScheme2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED, Grid>;

template class InternalScheme3D<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED, Grid>;


template class InternalScheme1D_ExHy<E_CENTERED, Grid>;
template class InternalScheme1D_ExHz<E_CENTERED, Grid>;
template class InternalScheme1D_EyHx<E_CENTERED, Grid>;
template class InternalScheme1D_EyHz<E_CENTERED, Grid>;
template class InternalScheme1D_EzHx<E_CENTERED, Grid>;
template class InternalScheme1D_EzHy<E_CENTERED, Grid>;

template class InternalScheme2D_TEx<E_CENTERED, Grid>;
template class InternalScheme2D_TEy<E_CENTERED, Grid>;
template class InternalScheme2D_TEz<E_CENTERED, Grid>;
template class InternalScheme2D_TMx<E_CENTERED, Grid>;
template class InternalScheme2D_TMy<E_CENTERED, Grid>;
template class InternalScheme2D_TMz<E_CENTERED, Grid>;

template class InternalScheme3D_3D<E_CENTERED, Grid>;


template class InternalScheme1D_ExHy_Grid<E_CENTERED>;
template class InternalScheme1D_ExHz_Grid<E_CENTERED>;
template class InternalScheme1D_EyHx_Grid<E_CENTERED>;
template class InternalScheme1D_EyHz_Grid<E_CENTERED>;
template class InternalScheme1D_EzHx_Grid<E_CENTERED>;
template class InternalScheme1D_EzHy_Grid<E_CENTERED>;

template class InternalScheme2D_TEx_Grid<E_CENTERED>;
template class InternalScheme2D_TEy_Grid<E_CENTERED>;
template class InternalScheme2D_TEz_Grid<E_CENTERED>;
template class InternalScheme2D_TMx_Grid<E_CENTERED>;
template class InternalScheme2D_TMy_Grid<E_CENTERED>;
template class InternalScheme2D_TMz_Grid<E_CENTERED>;

template class InternalScheme3D_3D_Grid<E_CENTERED>;

class INTERNAL_SCHEME_HELPER
{
public:

  HELPER_ALLOCATE_GRIDS
  HELPER_ALLOCATE_GRIDS_INC
  HELPER_ALLOCATE_GRIDS_FROM_CPU
  HELPER_ALLOCATE_GRIDS_ON_GPU

  HELPER_COPY_GRIDS_FROM_CPU
  HELPER_COPY_GRIDS_TO_GPU

  ICUDA_DEVICE
  static FieldValue approximateIncidentWaveHelper (FPValue d, IGRID<GridCoordinate1D> *FieldInc)
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
  ICUDA_DEVICE
  static
  FieldValue approximateIncidentWave (TCoord<FPValue, true>, TCoord<FPValue, true>, FPValue, IGRID<GridCoordinate1D> *, FPValue, FPValue);

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE ICUDA_HOST
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *EInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.0, EInc, incAngle1, incAngle2);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE ICUDA_HOST
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *HInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.5, HInc, incAngle1, incAngle2);
  }

#if defined (PARALLEL_GRID)
  HELPER_ALLOCATE_PARALLEL_GRIDS
#endif /* PARALLEL_GRID */
};

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class INTERNAL_SCHEME_BASE
{
  friend class INTERNAL_SCHEME_HELPER;
  INTERNAL_SCHEME_BASE_CPU_FRIEND
  INTERNAL_SCHEME_BASE_HELPER_CPU_FRIEND

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
  IGRID<TC> *Ex;
  IGRID<TC> *Ey;
  IGRID<TC> *Ez;
  IGRID<TC> *Hx;
  IGRID<TC> *Hy;
  IGRID<TC> *Hz;

  IGRID<TC> *Dx;
  IGRID<TC> *Dy;
  IGRID<TC> *Dz;
  IGRID<TC> *Bx;
  IGRID<TC> *By;
  IGRID<TC> *Bz;

  /**
   * Auxiliary field grids
   */
  IGRID<TC> *D1x;
  IGRID<TC> *D1y;
  IGRID<TC> *D1z;
  IGRID<TC> *B1x;
  IGRID<TC> *B1y;
  IGRID<TC> *B1z;

  /**
   * Amplitude field grids
   */
  IGRID<TC> *ExAmplitude;
  IGRID<TC> *EyAmplitude;
  IGRID<TC> *EzAmplitude;
  IGRID<TC> *HxAmplitude;
  IGRID<TC> *HyAmplitude;
  IGRID<TC> *HzAmplitude;

  /**
   * Material grids
   */
  IGRID<TC> *Eps;
  IGRID<TC> *Mu;

  /**
   * Sigmas
   */
  IGRID<TC> *SigmaX;
  IGRID<TC> *SigmaY;
  IGRID<TC> *SigmaZ;

  /**
   * Metamaterial grids
   */
  IGRID<TC> *OmegaPE;
  IGRID<TC> *GammaE;

  IGRID<TC> *OmegaPM;
  IGRID<TC> *GammaM;

  /**
   * Auxiliary TF/SF 1D grids
   */
  IGRID<GridCoordinate1D> *EInc;
  IGRID<GridCoordinate1D> *HInc;

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

#if defined (PARALLEL_GRID)
  ALLOCATE_PARALLEL_GRIDS
#endif

  ALLOCATE_GRIDS
  ALLOCATE_GRIDS_GPU
  COPY_GRIDS_GPU

  ICUDA_HOST
  virtual void initCoordTypes () { ALWAYS_ASSERT (0); }

  ICUDA_DEVICE ICUDA_HOST
  virtual bool doSkipBorderFunc (TC, IGRID<TC> *) { ALWAYS_ASSERT (0); return false; }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { ALWAYS_ASSERT (0); }
#endif /* ENABLE_ASSERTS */

  template <uint8_t grid_type>
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &, TC, TC, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStep (time_step, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepInit (IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *);

  template <uint8_t grid_type, bool usePML>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIteration (time_step, TC, IGRID<TC> *, GridType, IGRID<TC> *, GridType, IGRID<TC> *, IGRID<TC> *, SourceCallBack, FPValue);

  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <bool useMetamaterials>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationPML (time_step, TC, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationBorder (time_step, TC, IGRID<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationExact (time_step, TC, IGRID<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);

  template<uint8_t EnumVal>
  ICUDA_DEVICE
  void performPointSourceCalc (time_step);

  ICUDA_DEVICE ICUDA_HOST
  FieldValue calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                        FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                        FPValue Ca, FPValue Cb, FPValue delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return prev * Ca + tmp * Cb;
  }

  ICUDA_DEVICE ICUDA_HOST
  FieldValue calcFieldDrude (FieldValue curDOrB, FieldValue prevDOrB, FieldValue prevPrevDOrB,
                             FieldValue prevEOrH, FieldValue prevPrevEOrH,
                             FPValue b0, FPValue b1, FPValue b2, FPValue a1, FPValue a2)
  {
    return curDOrB * b0 + prevDOrB * b1 + prevPrevDOrB * b2 - prevEOrH * a1 - prevPrevEOrH * a2;
  }

  ICUDA_DEVICE ICUDA_HOST
  FieldValue calcFieldFromDOrB (FieldValue prevEOrH, FieldValue curDOrB, FieldValue prevDOrB,
                                FPValue Ca, FPValue Cb, FPValue Cc)
  {
    return prevEOrH * Ca + curDOrB * Cb - prevDOrB * Cc;
  }

public:

  ICUDA_HOST
  INTERNAL_SCHEME_BASE ();

  ICUDA_HOST
  virtual ~INTERNAL_SCHEME_BASE ();

  INIT
  INIT_FROM_CPU
  INIT_ON_GPU

  COPY_FROM_CPU
  COPY_TO_GPU

  COPY_BACK_TO_CPU

  ICUDA_HOST
  void
  initScheme (FPValue, FPValue);

  PERFORM_FIELD_STEPS_KERNEL
  SHIFT_IN_TIME_KERNEL_LAUNCHES
  PERFORM_PLANE_WAVE_STEPS_KERNELS
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCHES

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void performFieldSteps (time_step t, TC Start, TC End)
  {
    TC zero (0, 0, 0
#ifdef DEBUG_INFO
             , ct1, ct2, ct3
#endif /* DEBUG_INFO */
             );

    TC start;
    TC end;

    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        start = Ex->getComputationStart (zero);
        end = Ex->getComputationEnd (zero);
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        start = Ey->getComputationStart (zero);
        end = Ey->getComputationEnd (zero);
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        start = Ez->getComputationStart (zero);
        end = Ez->getComputationEnd (zero);
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        start = Hx->getComputationStart (zero);
        end = Hx->getComputationEnd (zero);
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        start = Hy->getComputationStart (zero);
        end = Hy->getComputationEnd (zero);
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        start = Hz->getComputationStart (zero);
        end = Hz->getComputationEnd (zero);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (Start < start)
    {
      Start = start;
    }
    if (End > end)
    {
      End = end;
    }

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

  ICUDA_DEVICE
  void performPlaneWaveESteps (time_step, GridCoordinate1D start, GridCoordinate1D end);
  ICUDA_DEVICE
  void performPlaneWaveHSteps (time_step, GridCoordinate1D start, GridCoordinate1D end);

  ICUDA_DEVICE ICUDA_HOST
  FieldValue approximateIncidentWaveE (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveE<Type, TCoord> (pos, layout->getZeroIncCoordFP (), EInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  FieldValue approximateIncidentWaveH (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveH<Type, TCoord> (pos, layout->getZeroIncCoordFP (), HInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }

  ICUDA_DEVICE
  FPValue getMaterial (const TC &, GridType, IGRID<TC> *, GridType);
  ICUDA_DEVICE
  FPValue getMetaMaterial (const TC &, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType,
                           FPValue &, FPValue &);

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEx () const { return doNeedEx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEy () const { return doNeedEy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEz () const { return doNeedEz; }

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHx () const { return doNeedHx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHy () const { return doNeedHy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHz () const { return doNeedHz; }

  CoordinateType getType1 ()
  {
    return ct1;
  }
  CoordinateType getType2 ()
  {
    return ct2;
  }
  CoordinateType getType3 ()
  {
    return ct3;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEx ()
  {
    ASSERT (Ex);
    ASSERT (doNeedEx);
    return Ex;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEy ()
  {
    ASSERT (Ey);
    ASSERT (doNeedEy);
    return Ey;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEz ()
  {
    ASSERT (Ez);
    ASSERT (doNeedEz);
    return Ez;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHx ()
  {
    ASSERT (Hx);
    ASSERT (doNeedHx);
    return Hx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHy ()
  {
    ASSERT (Hy);
    ASSERT (doNeedHy);
    return Hy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHz ()
  {
    ASSERT (Hz);
    ASSERT (doNeedHz);
    return Hz;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEps ()
  {
    ASSERT (Eps);
    return Eps;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getMu ()
  {
    ASSERT (Mu);
    return Mu;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaX ()
  {
    ASSERT (SigmaX);
    ASSERT (doNeedSigmaX);
    return SigmaX;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaY ()
  {
    ASSERT (SigmaY);
    ASSERT (doNeedSigmaY);
    return SigmaY;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaZ ()
  {
    ASSERT (SigmaZ);
    ASSERT (doNeedSigmaZ);
    return SigmaZ;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getOmegaPE ()
  {
    ASSERT (OmegaPE);
    return OmegaPE;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getOmegaPM ()
  {
    ASSERT (OmegaPM);
    return OmegaPM;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getGammaE ()
  {
    ASSERT (GammaE);
    return GammaE;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getGammaM ()
  {
    ASSERT (GammaM);
    return GammaM;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDx ()
  {
    ASSERT (Dx);
    ASSERT (doNeedEx);
    return Dx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDy ()
  {
    ASSERT (Dy);
    ASSERT (doNeedEy);
    return Dy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDz ()
  {
    ASSERT (Dz);
    ASSERT (doNeedEz);
    return Dz;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBx ()
  {
    ASSERT (Bx);
    ASSERT (doNeedHx);
    return Bx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBy ()
  {
    ASSERT (By);
    ASSERT (doNeedHy);
    return By;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBz ()
  {
    ASSERT (Bz);
    ASSERT (doNeedHz);
    return Bz;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1x ()
  {
    ASSERT (D1x);
    ASSERT (doNeedEx);
    return D1x;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1y ()
  {
    ASSERT (D1y);
    ASSERT (doNeedEy);
    return D1y;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1z ()
  {
    ASSERT (D1z);
    ASSERT (doNeedEz);
    return D1z;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1x ()
  {
    ASSERT (B1x);
    ASSERT (doNeedHx);
    return B1x;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1y ()
  {
    ASSERT (B1y);
    ASSERT (doNeedHy);
    return B1y;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1z ()
  {
    ASSERT (B1z);
    ASSERT (doNeedHz);
    return B1z;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<GridCoordinate1D> * getEInc ()
  {
    ASSERT (EInc);
    return EInc;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<GridCoordinate1D> * getHInc ()
  {
    ASSERT (HInc);
    return HInc;
  }
};

/*
 * Dimension specific
 */

template <SchemeType_t Type, LayoutType layout_type>
class INTERNAL_SCHEME_1D: public INTERNAL_SCHEME_BASE<Type, GridCoordinate1DTemplate, layout_type>
{
protected:

#ifdef PARALLEL_GRID
  ALLOCATE_PARALLEL_GRIDS_OVERRIDE
#endif /* PARALLEL_GRID */

  ICUDA_DEVICE ICUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate1D pos, IGRID<GridCoordinate1D> *grid) CXX11_OVERRIDE_FINAL
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1;
  }
};

template <SchemeType_t Type, LayoutType layout_type>
class INTERNAL_SCHEME_2D: public INTERNAL_SCHEME_BASE<Type, GridCoordinate2DTemplate, layout_type>
{
protected:

#ifdef PARALLEL_GRID
  ALLOCATE_PARALLEL_GRIDS_OVERRIDE
#endif /* PARALLEL_GRID */

  ICUDA_DEVICE ICUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate2D pos, IGRID<GridCoordinate2D> *grid) CXX11_OVERRIDE_FINAL
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1;
  }
};

template <SchemeType_t Type, LayoutType layout_type>
class INTERNAL_SCHEME_3D: public INTERNAL_SCHEME_BASE<Type, GridCoordinate3DTemplate, layout_type>
{
protected:

#ifdef PARALLEL_GRID
  ALLOCATE_PARALLEL_GRIDS_OVERRIDE
#endif /* PARALLEL_GRID */

  ICUDA_DEVICE ICUDA_HOST
  virtual bool doSkipBorderFunc (GridCoordinate3D pos, IGRID<GridCoordinate3D> *grid) CXX11_OVERRIDE_FINAL
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1
           && pos.get3 () != 0 && pos.get3 () != grid->getTotalSize ().get3 () - 1;
  }
};

/*
 * Scheme type specific
 */

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EX_HY: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EX_HZ: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EY_HX: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EY_HZ: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EZ_HX: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_1D_EZ_HY: public INTERNAL_SCHEME_1D<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type>::ct2 = CoordinateType::NONE;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TEX: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TEY: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TEZ: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TMX: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TMY: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Z;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_2D_TMZ: public INTERNAL_SCHEME_2D<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type>::ct2 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type>::ct3 = CoordinateType::NONE;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
  }
#endif /* ENABLE_ASSERTS */
};

template <LayoutType layout_type>
class INTERNAL_SCHEME_3D_3D: public INTERNAL_SCHEME_3D<(static_cast<SchemeType_t> (SchemeType::Dim3)), layout_type>
{
protected:

  ICUDA_HOST
  virtual void initCoordTypes () CXX11_OVERRIDE_FINAL
  {
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type>::ct1 = CoordinateType::X;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type>::ct2 = CoordinateType::Y;
    INTERNAL_SCHEME_BASE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type>::ct3 = CoordinateType::Z;
  }

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFExAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () < pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEyAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () < pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFEzAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHxAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () == pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () < pos22.get2 ());
    ASSERT (pos11.get3 () < pos12.get3 ());
    ASSERT (pos21.get3 () == pos22.get3 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  virtual void calculateTFSFHyAsserts (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22) CXX11_OVERRIDE_FINAL
  {
    ASSERT (pos11.get1 () < pos12.get1 ());
    ASSERT (pos21.get1 () == pos22.get1 ());
    ASSERT (pos11.get2 () == pos12.get2 ());
    ASSERT (pos21.get2 () == pos22.get2 ());
    ASSERT (pos11.get3 () == pos12.get3 ());
    ASSERT (pos21.get3 () < pos22.get3 ());
  }
  ICUDA_DEVICE ICUDA_HOST
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

#include "InternalScheme.template.inc.h"

#undef INTERNAL_SCHEME_BASE
#undef INTERNAL_SCHEME_BASE_CPU_FRIEND
#undef INTERNAL_SCHEME_BASE_HELPER_CPU_FRIEND
#undef INTERNAL_SCHEME_HELPER
#undef IGRID
#undef ICUDA_HOST
#undef ICUDA_DEVICE

#undef ALLOCATE_PARALLEL_GRIDS
#undef ALLOCATE_PARALLEL_GRIDS_OVERRIDE

#undef INIT
#undef INIT_FROM_CPU
#undef INIT_ON_GPU
#undef COPY_FROM_CPU
#undef COPY_TO_GPU
#undef COPY_BACK_TO_CPU

#undef INTERNAL_SCHEME_1D
#undef INTERNAL_SCHEME_2D
#undef INTERNAL_SCHEME_3D

#undef INTERNAL_SCHEME_1D_EX_HY
#undef INTERNAL_SCHEME_1D_EX_HZ
#undef INTERNAL_SCHEME_1D_EY_HX
#undef INTERNAL_SCHEME_1D_EY_HZ
#undef INTERNAL_SCHEME_1D_EZ_HX
#undef INTERNAL_SCHEME_1D_EZ_HY
#undef INTERNAL_SCHEME_2D_TEX
#undef INTERNAL_SCHEME_2D_TEY
#undef INTERNAL_SCHEME_2D_TEZ
#undef INTERNAL_SCHEME_2D_TMX
#undef INTERNAL_SCHEME_2D_TMY
#undef INTERNAL_SCHEME_2D_TMZ
#undef INTERNAL_SCHEME_3D_3D

#undef ALLOCATE_GRIDS
#undef ALLOCATE_GRIDS_GPU
#undef COPY_GRIDS_GPU
#undef HELPER_ALLOCATE_GRIDS
#undef HELPER_ALLOCATE_GRIDS_INC
#undef HELPER_ALLOCATE_GRIDS_FROM_CPU
#undef HELPER_ALLOCATE_GRIDS_ON_GPU
#undef HELPER_COPY_GRIDS_FROM_CPU
#undef HELPER_COPY_GRIDS_TO_GPU
#undef HELPER_ALLOCATE_PARALLEL_GRIDS

#undef PERFORM_FIELD_STEPS_KERNEL
#undef PERFORM_PLANE_WAVE_STEPS_KERNELS
#undef SHIFT_IN_TIME_KERNEL_LAUNCHES
#undef SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCHES

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateTFSF (TC posAbs,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepInitDiff (TCS *diff11, TCS *diff12, TCS *diff21, TCS *diff22)
{
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      *diff11 = yeeLayout->getExCircuitElementDiff (LayoutDirection::DOWN);
      *diff12 = yeeLayout->getExCircuitElementDiff (LayoutDirection::UP);
      *diff21 = yeeLayout->getExCircuitElementDiff (LayoutDirection::BACK);
      *diff22 = yeeLayout->getExCircuitElementDiff (LayoutDirection::FRONT);
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      *diff11 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::BACK);
      *diff12 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::FRONT);
      *diff21 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::LEFT);
      *diff22 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::RIGHT);
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      *diff11 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::LEFT);
      *diff12 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::RIGHT);
      *diff21 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::DOWN);
      *diff22 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::UP);
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      *diff11 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::BACK);
      *diff12 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::FRONT);
      *diff21 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::DOWN);
      *diff22 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::UP);
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      *diff11 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::LEFT);
      *diff12 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::RIGHT);
      *diff21 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::BACK);
      *diff22 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::FRONT);
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      *diff11 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::DOWN);
      *diff12 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::UP);
      *diff21 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::LEFT);
      *diff22 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::RIGHT);
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

/**
 * Initialize grids used in further computations
 *
 * TODO: force inline this
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepInit (IGRID<TC> **grid, GridType *gridType, IGRID<TC> **materialGrid, GridType *materialGridType, IGRID<TC> **materialGrid1, GridType *materialGridType1,
IGRID<TC> **materialGrid2, GridType *materialGridType2, IGRID<TC> **materialGrid3, GridType *materialGridType3, IGRID<TC> **materialGrid4, GridType *materialGridType4,
IGRID<TC> **materialGrid5, GridType *materialGridType5, IGRID<TC> **oppositeGrid1, IGRID<TC> **oppositeGrid2, IGRID<TC> **gridPML1, GridType *gridPMLType1, IGRID<TC> **gridPML2, GridType *gridPMLType2,
SourceCallBack *rightSideFunc, SourceCallBack *borderFunc, SourceCallBack *exactFunc, FPValue *materialModifier,
  IGRID<TC> **Ca, IGRID<TC> **Cb, IGRID<TC> **CB0, IGRID<TC> **CB1, IGRID<TC> **CB2, IGRID<TC> **CA1, IGRID<TC> **CA2, IGRID<TC> **CaPML, IGRID<TC> **CbPML, IGRID<TC> **CcPML)
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

      *Ca = CaEx;
      *Cb = CbEx;

      *CB0 = CB0Ex;
      *CB1 = CB1Ex;
      *CB2 = CB2Ex;

      *CaPML = CaPMLEx;
      *CbPML = CbPMLEx;
      *CcPML = CcPMLEx;

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

      *Ca = CaEy;
      *Cb = CbEy;

      *CB0 = CB0Ey;
      *CB1 = CB1Ey;
      *CB2 = CB2Ey;

      *CaPML = CaPMLEy;
      *CbPML = CbPMLEy;
      *CcPML = CcPMLEy;

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

      *Ca = CaEz;
      *Cb = CbEz;

      *CB0 = CB0Ez;
      *CB1 = CB1Ez;
      *CB2 = CB2Ez;

      *CaPML = CaPMLEz;
      *CbPML = CbPMLEz;
      *CcPML = CcPMLEz;

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

      *Ca = DaHx;
      *Cb = DbHx;

      *CB0 = DB0Hx;
      *CB1 = DB1Hx;
      *CB2 = DB2Hx;

      *CaPML = DaPMLHx;
      *CbPML = DbPMLHx;
      *CcPML = DcPMLHx;

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

      *Ca = DaHy;
      *Cb = DbHy;

      *CB0 = DB0Hy;
      *CB1 = DB1Hy;
      *CB2 = DB2Hy;

      *CaPML = DaPMLHy;
      *CbPML = DbPMLHy;
      *CcPML = DcPMLHy;

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

      *Ca = DaHz;
      *Cb = DbHz;

      *CB0 = DB0Hz;
      *CB1 = DB1Hz;
      *CB2 = DB2Hz;

      *CaPML = DaPMLHz;
      *CbPML = DbPMLHz;
      *CcPML = DcPMLHz;

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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIteration (time_step t,
                                                                             TC pos,
                                                                             TC posAbs,
                                                                             TCS diff11,
                                                                             TCS diff12,
                                                                             TCS diff21,
                                                                             TCS diff22,
                                                                             IGRID<TC> *grid,
                                                                             TCFP coordFP,
                                                                             IGRID<TC> *oppositeGrid1,
                                                                             IGRID<TC> *oppositeGrid2,
                                                                             SourceCallBack rightSideFunc,
                                                                             IGRID<TC> *Ca,
                                                                             IGRID<TC> *Cb,
                                                                             bool usePML,
                                                                             GridType gridType,
                                                                             IGRID<TC> *materialGrid,
                                                                             GridType materialGridType,
                                                                             FPValue materialModifier)
{
  // TODO: [possible] move 1D gridValues to 3D gridValues array
  ASSERT (grid != NULLPTR);
  grid_coord coord = grid->calculateIndexFromPosition (pos);
  FieldValue val = *grid->getFieldValue (coord, 1);

  FieldValue valCa = FIELDVALUE (0, 0);
  FieldValue valCb = FIELDVALUE (0, 0);

  if (usePrecomputedGrids)
  {
    ASSERT (Ca != NULLPTR);
    ASSERT (Cb != NULLPTR);
    valCa = *Ca->getFieldValue (pos, 0);
    valCb = *Cb->getFieldValue (pos, 0);
  }
  else
  {
    ASSERT (Ca == NULLPTR);
    ASSERT (Cb == NULLPTR);
    ASSERT (materialGrid != NULLPTR);

    FPValue material = getMaterial (posAbs, gridType, materialGrid, materialGridType);
    FPValue ca = FPValue (0);
    FPValue cb = FPValue (0);

    FPValue k_mod = FPValue (1);

    if (usePML)
    {
      FPValue eps0 = PhysicsConst::Eps0;
      ca = (2 * eps0 * k_mod - material * gridTimeStep) / (2 * eps0 * k_mod + material * gridTimeStep);
      cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_mod + material * gridTimeStep);
    }
    else
    {
      ca = 1.0;
      cb = gridTimeStep / (material * materialModifier * gridStep);
    }

    valCa = FIELDVALUE (ca, 0);
    valCb = FIELDVALUE (cb, 0);
  }

  ASSERT (valCa != FIELDVALUE (0, 0));
  ASSERT (valCb != FIELDVALUE (0, 0));

  FieldValue prev11 = FIELDVALUE (0, 0);
  FieldValue prev12 = FIELDVALUE (0, 0);
  FieldValue prev21 = FIELDVALUE (0, 0);
  FieldValue prev22 = FIELDVALUE (0, 0);

  FieldValue prevRightSide = FIELDVALUE (0, 0);

  if (oppositeGrid1)
  {
    prev11 = *oppositeGrid1->getFieldValue (pos + diff11, 1);
    prev12 = *oppositeGrid1->getFieldValue (pos + diff12, 1);
  }

  if (oppositeGrid2)
  {
    prev21 = *oppositeGrid2->getFieldValue (pos + diff21, 1);
    prev22 = *oppositeGrid2->getFieldValue (pos + diff22, 1);
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    calculateTFSF<grid_type> (posAbs, prev11, prev12, prev21, prev22, pos + diff11, pos + diff12, pos + diff21, pos + diff22);
  }

  if (rightSideFunc != NULLPTR)
  {
    prevRightSide = rightSideFunc (expandTo3D (coordFP * gridStep, ct1, ct2, ct3), t * gridTimeStep);
  }

  FieldValue valNew = calcField (val, prev12, prev11, prev22, prev21, prevRightSide, valCa, valCb, gridStep);
  grid->setFieldValue (valNew, coord, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPMLMetamaterials (time_step t,
                                                                               TC pos,
                                                                               IGRID<TC> *grid,
                                                                               IGRID<TC> *gridPML,
                                                                               IGRID<TC> *CB0,
                                                                               IGRID<TC> *CB1,
                                                                               IGRID<TC> *CB2,
                                                                               IGRID<TC> *CA1,
                                                                               IGRID<TC> *CA2,
                                                                               GridType gridType,
                                                                               IGRID<TC> *materialGrid1,
                                                                               GridType materialGridType1,
                                                                               IGRID<TC> *materialGrid2,
                                                                               GridType materialGridType2,
                                                                               IGRID<TC> *materialGrid3,
                                                                               GridType materialGridType3,
                                                                               FPValue materialModifier)
{
  ASSERT (grid != NULLPTR);
  ASSERT (gridPML != NULLPTR);
  grid_coord coord = grid->calculateIndexFromPosition (pos);

  FieldValue cur = *grid->getFieldValue (coord, 0);
  FieldValue prev = *grid->getFieldValue (coord, 1);
  FieldValue prevPrev = *grid->getFieldValue (coord, 2);

  FieldValue prevPML = *gridPML->getFieldValue (coord, 1);
  FieldValue prevPrevPML = *gridPML->getFieldValue (coord, 2);

  FieldValue valb0 = FIELDVALUE (0, 0);
  FieldValue valb1 = FIELDVALUE (0, 0);
  FieldValue valb2 = FIELDVALUE (0, 0);
  FieldValue vala1 = FIELDVALUE (0, 0);
  FieldValue vala2 = FIELDVALUE (0, 0);

  if (usePrecomputedGrids)
  {
    ASSERT (CB0 != NULLPTR);
    ASSERT (CB1 != NULLPTR);
    ASSERT (CB2 != NULLPTR);
    ASSERT (CA1 != NULLPTR);
    ASSERT (CA2 != NULLPTR);

    valb0 = *CB0->getFieldValue (coord, 0);
    valb1 = *CB1->getFieldValue (coord, 0);
    valb2 = *CB2->getFieldValue (coord, 0);
    vala1 = *CA1->getFieldValue (coord, 0);
    vala2 = *CA2->getFieldValue (coord, 0);
  }
  else
  {
    ASSERT (CB0 == NULLPTR);
    ASSERT (CB1 == NULLPTR);
    ASSERT (CB2 == NULLPTR);
    ASSERT (CA1 == NULLPTR);
    ASSERT (CA2 == NULLPTR);

    ASSERT (materialGrid1 != NULLPTR);
    ASSERT (materialGrid2 != NULLPTR);
    ASSERT (materialGrid3 != NULLPTR);

    TC posAbs = grid->getTotalPosition (pos);

    FPValue material1;
    FPValue material2;

    FPValue material = getMetaMaterial (posAbs, gridType,
                                        materialGrid1, materialGridType1,
                                        materialGrid2, materialGridType2,
                                        materialGrid3, materialGridType3,
                                        material1, material2);

    FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
    FPValue b0 = (4 + 2*gridTimeStep*material2) / A;
    FPValue b1 = -8 / A;
    FPValue b2 = (4 - 2*gridTimeStep*material2) / A;
    FPValue a1 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
    FPValue a2 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;

    valb0 = FIELDVALUE (b0, 0);
    valb1 = FIELDVALUE (b1, 0);
    valb2 = FIELDVALUE (b2, 0);
    vala1 = FIELDVALUE (a1, 0);
    vala2 = FIELDVALUE (a2, 0);
  }

  ASSERT (valb0 != FIELDVALUE (0, 0));
  ASSERT (valb1 != FIELDVALUE (0, 0));
  ASSERT (valb2 != FIELDVALUE (0, 0));
  ASSERT (vala1 != FIELDVALUE (0, 0));
  ASSERT (vala2 != FIELDVALUE (0, 0));

  FieldValue valNew = calcFieldDrude (cur, prev, prevPrev, prevPML, prevPrevPML, valb0, valb1, valb2, vala1, vala2);
  gridPML->setFieldValue (valNew, coord, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool useMetamaterials, bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPML (time_step t,
                                                                   TC pos,
                                                                   IGRID<TC> *grid,
                                                                   IGRID<TC> *gridPML1,
                                                                   IGRID<TC> *gridPML2,
                                                                   IGRID<TC> *Ca,
                                                                   IGRID<TC> *Cb,
                                                                   IGRID<TC> *Cc,
                                                                   GridType gridPMLType1,
                                                                   IGRID<TC> *materialGrid1,
                                                                   GridType materialGridType1,
                                                                   IGRID<TC> *materialGrid4,
                                                                   GridType materialGridType4,
                                                                   IGRID<TC> *materialGrid5,
                                                                   GridType materialGridType5,
                                                                   FPValue materialModifier)
{
  ASSERT (grid != NULLPTR);
  ASSERT (gridPML1 != NULLPTR);
  ASSERT (gridPML2 != NULLPTR);
  grid_coord coord = grid->calculateIndexFromPosition (pos);

  FieldValue prevEorH = *gridPML2->getFieldValue (coord, 1);
  FieldValue curDorB = FIELDVALUE (0, 0);
  FieldValue prevDorB = FIELDVALUE (0, 0);

  FieldValue valCa = FIELDVALUE (0, 0);
  FieldValue valCb = FIELDVALUE (0, 0);
  FieldValue valCc = FIELDVALUE (0, 0);

  if (useMetamaterials)
  {
    curDorB = gridPML1->getFieldValue (coord, 0);
    prevDorB = gridPML1->getFieldValue (coord, 1);
  }
  else
  {
    curDorB = grid->getFieldValue (coord, 0);
    prevDorB = grid->getFieldValue (coord, 1);
  }

  if (usePrecomputedGrids)
  {
    ASSERT (Ca != NULLPTR);
    ASSERT (Cb != NULLPTR);
    ASSERT (Cc != NULLPTR);

    valCa = *Ca->getFieldValue (coord, 0);
    valCb = *Cb->getFieldValue (coord, 0);
    valCc = *Cc->getFieldValue (coord, 0);
  }
  else
  {
    FPValue eps0 = PhysicsConst::Eps0;
    TC posAbs = gridPML2->getTotalPosition (pos);

    FPValue material1 = materialGrid1 ? getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1) : 0;
    FPValue material4 = materialGrid4 ? getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4) : 0;
    FPValue material5 = materialGrid5 ? getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5) : 0;

    FPValue modifier = material1 * materialModifier;
    if (useMetamaterials)
    {
      modifier = 1;
    }

    FPValue k_mod1 = 1;
    FPValue k_mod2 = 1;

    FPValue ca = (2 * eps0 * k_mod2 - material5 * gridTimeStep) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
    FPValue cb = ((2 * eps0 * k_mod1 + material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
    FPValue cc = ((2 * eps0 * k_mod1 - material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);

    valCa = FIELDVALUE (ca, 0);
    valCb = FIELDVALUE (cb, 0);
    valCc = FIELDVALUE (cc, 0);
  }

  ASSERT (Ca != FIELDVALUE (0, 0));
  ASSERT (Cb != FIELDVALUE (0, 0));
  ASSERT (Cc != FIELDVALUE (0, 0));

  FieldValue valNew = calcFieldFromDOrB (prevEorH, curDorB, prevDorB, valCa, valCb, valCc);
  gridPML2->setFieldValue (valNew, coord, 0);
}

#ifndef GPU_INTERNAL_SCHEME

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationBorder (time_step t,
                                                                      TC pos,
                                                                      IGRID<TC> *grid,
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

  grid->setFieldValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep), pos, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationExact (time_step t,
                                                                     TC pos,
                                                                     IGRID<TC> *grid,
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

  FieldValue numerical = *grid->getFieldValue (pos, 0);
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

#endif /* !GPU_INTERNAL_SCHEME */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t EnumVal>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPointSourceCalc (time_step t)
{
  IGRID<TC> *grid = NULLPTR;

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

  FieldValue* pointVal = grid->getFieldValueOrNullByAbsolutePos (pos, 0);

  if (pointVal)
  {
#ifdef COMPLEX_FIELD_VALUES
    *pointVal = FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                            cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#else /* COMPLEX_FIELD_VALUES */
    *pointVal = setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
FPValue
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::getMaterial (const TC &posAbs,
                            GridType typeOfField,
                            IGRID<TC> *gridMaterial,
                            GridType typeOfMaterial)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  yeeLayout->template initCoordinates<false> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                          absPos31, absPos32, absPos41, absPos42);

  ASSERT (typeOfMaterial == GridType::EPS
          || typeOfMaterial == GridType::MU
          || typeOfMaterial == GridType::SIGMAX
          || typeOfMaterial == GridType::SIGMAY
          || typeOfMaterial == GridType::SIGMAZ);

  if (yeeLayout->getIsDoubleMaterialPrecision ())
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos31, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos32, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos41, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos42, 0)).real ();
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::EZ:
      case GridType::DZ:
      {
        return yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0)).real ();
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0)).real ();
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  UNREACHABLE;
  return FPValue (0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
FPValue
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::getMetaMaterial (const TC &posAbs,
                                GridType typeOfField,
                                IGRID<TC> *gridMaterial,
                                GridType typeOfMaterial,
                                IGRID<TC> *gridMaterialOmega,
                                GridType typeOfMaterialOmega,
                                IGRID<TC> *gridMaterialGamma,
                                GridType typeOfMaterialGamma,
                                FPValue &omega,
                                FPValue &gamma)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  yeeLayout->template initCoordinates<true> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                         absPos31, absPos32, absPos41, absPos42);

  ASSERT ((typeOfMaterialOmega == GridType::OMEGAPE && typeOfMaterialGamma == GridType::GAMMAE)
          || (typeOfMaterialOmega == GridType::OMEGAPM && typeOfMaterialGamma == GridType::GAMMAM));

  if (yeeLayout->getIsDoubleMaterialPrecision ())
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos42, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos42, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos42, 0),
                                                      omega, gamma).real ();
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::EZ:
      case GridType::DZ:
      {
        return yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                      omega, gamma).real ();
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos22, 0),
                                                      omega, gamma).real ();
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  UNREACHABLE;
  return FPValue (0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::INTERNAL_SCHEME_BASE ()
  : isInitialized (false)
  , useParallel (false)
  , yeeLayout (NULLPTR)

#define GRID_NAME(x) \
  , x (NULLPTR)
#include "Grids.inc.h"
#undef GRID_NAME

  , EInc (NULLPTR)
  , HInc (NULLPTR)
  , sourceWaveLength (0)
  , sourceWaveLengthNumerical (0)
  , sourceFrequency (0)
  , courantNum (0)
  , gridStep (0)
  , gridTimeStep (0)
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPlaneWaveESteps (time_step t, GridCoordinate1D start, GridCoordinate1D end)
{
  ASSERT (end.get1 () > start.get1 ());
  ASSERT (end.get1 () <= EInc->getSize ().get1 ());

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Eps0 * gridStep);

  grid_coord cstart = start.get1 ();
  grid_coord cend = end.get1 ();

  bool setSource = false;
  if (cstart == 0)
  {
    setSource = true;
    cstart = 1;
  }

  for (grid_coord i = cstart; i < cend; ++i)
  {
    FieldValue valE = *EInc->getFieldValue (i, 1);
    FieldValue valH1 = *HInc->getFieldValue (i - 1, 1);
    FieldValue valH2 = *HInc->getFieldValue (i, 1);

    FieldValue val = valE + (valH1 - valH2) * modifier;
    EInc->setFieldValue (val, i, 0);
  }

  if (setSource)
  {
    FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;

#ifdef COMPLEX_FIELD_VALUES
    EInc->setFieldValue (FieldValue (sin (arg), cos (arg)), 0, 0);
#else /* COMPLEX_FIELD_VALUES */
    EInc->setFieldValue (sin (arg), 0, 0);
#endif /* !COMPLEX_FIELD_VALUES */

    //printf ("EInc[0] %f \n", valE->getCurValue ());
  }

#ifdef ENABLE_ASSERTS
  ALWAYS_ASSERT (*EInc->getFieldValue (EInc->getSize ().get1 () - 1, 0) == getFieldValueRealOnly (0.0));
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPlaneWaveHSteps (time_step t, GridCoordinate1D start, GridCoordinate1D end)
{
  ASSERT (end.get1 () > start.get1 ());
  ASSERT (end.get1 () <= HInc->getSize ().get1 ());

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Mu0 * gridStep);

  grid_coord cstart = start.get1 ();
  grid_coord cend = end.get1 ();

  if (cend == HInc->getSize ().get1 ())
  {
    cend--;
  }

  for (grid_coord i = cstart; i < cend; ++i)
  {
    FieldValue valH = *HInc->getFieldValue (i, 1);
    FieldValue valE1 = *EInc->getFieldValue (i, 1);
    FieldValue valE2 = *EInc->getFieldValue (i + 1, 1);

    FieldValue val = valH + (valE1 - valE2) * modifier;
    HInc->setFieldValue (val, i, 0);
  }

#ifdef ENABLE_ASSERTS
  ALWAYS_ASSERT (*HInc->getFieldValue (HInc->getSize ().get1 () - 2, 0) == getFieldValueRealOnly (0.0));
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::initScheme (FPValue dx, FPValue sourceWaveLen)
{
  sourceWaveLength = sourceWaveLen;
  sourceFrequency = PhysicsConst::SpeedOfLight / sourceWaveLength;

  gridStep = dx;
  courantNum = SOLVER_SETTINGS.getCourantNum ();
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  ALWAYS_ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

  FPValue phaseVelocity0 = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, PhysicsConst::Pi / 2, 0);
  FPValue phaseVelocity = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ());
  FPValue k = 2 * PhysicsConst::Pi * PhysicsConst::SpeedOfLight / sourceWaveLength / phaseVelocity0;

  relPhaseVelocity = phaseVelocity0 / phaseVelocity;
  sourceWaveLengthNumerical = 2 * PhysicsConst::Pi / k;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "initScheme: "
                                      "\n\tphase velocity relation -> %f "
                                      "\n\tphase velosity 0 -> %f "
                                      "\n\tphase velocity -> %f "
                                      "\n\tanalytical wave number -> %.20f "
                                      "\n\tnumerical wave number -> %.20f"
                                      "\n\tanalytical wave length -> %.20f"
                                      "\n\tnumerical wave length -> %.20f"
                                      "\n\tnumerical grid step -> %.20f"
                                      "\n\tnumerical time step -> %.20f"
                                      "\n\twave length -> %.20f"
                                      "\n",
           relPhaseVelocity, phaseVelocity0, phaseVelocity, 2*PhysicsConst::Pi/sourceWaveLength, k,
           sourceWaveLength, sourceWaveLengthNumerical, gridStep, gridTimeStep, sourceFrequency);
}

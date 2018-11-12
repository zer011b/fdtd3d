template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE ICUDA_HOST
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
  IGRID<TC> **Ca, IGRID<TC> **Cb)
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
template<uint8_t grid_type, bool usePML>
ICUDA_DEVICE ICUDA_HOST
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
                                                               IGRID<TC> *Cb)
{
  // TODO: [possible] move 1D gridValues to 3D gridValues array
  grid_coord coord = grid->calculateIndexFromPosition (pos);
  FieldValue val = *grid->getFieldValue (coord, 1);

  FieldValue valCa = *Ca->getFieldValue (pos, 0);
  FieldValue valCb = *Cb->getFieldValue (pos, 0);

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

// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// ICUDA_DEVICE ICUDA_HOST
// void
// INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPMLMetamaterials (time_step t,
//                                                                                TC pos,
//                                                                                IGRID<TC> *grid,
//                                                                                IGRID<TC> *gridPML,
//                                                                                GridType gridType,
//                                                                                IGRID<TC> *materialGrid1,
//                                                                                GridType materialGridType1,
//                                                                                IGRID<TC> *materialGrid2,
//                                                                                GridType materialGridType2,
//                                                                                IGRID<TC> *materialGrid3,
//                                                                                GridType materialGridType3,
//                                                                                FPValue materialModifier)
// {
//   TC posAbs = grid->getTotalPosition (pos);
//   FieldPointValue *valField = grid->getFieldPointValue (pos);
//   FieldPointValue *valField1 = gridPML->getFieldPointValue (pos);
//
//   FPValue material1;
//   FPValue material2;
//
//   FPValue material = getMetaMaterial (posAbs, gridType,
//                                                  materialGrid1, materialGridType1,
//                                                  materialGrid2, materialGridType2,
//                                                  materialGrid3, materialGridType3,
//                                                  material1, material2);
//
//   /*
//    * TODO: precalculate coefficients
//    */
//   FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
//   FPValue a1 = (4 + 2*gridTimeStep*material2) / A;
//   FPValue a2 = -8 / A;
//   FPValue a3 = (4 - 2*gridTimeStep*material2) / A;
//   FPValue a4 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
//   FPValue a5 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;
//
// #if defined (TWO_TIME_STEPS)
//   FieldValue val = calcFieldDrude (valField->getCurValue (),
//                                    valField->getPrevValue (),
//                                    valField->getPrevPrevValue (),
//                                    valField1->getPrevValue (),
//                                    valField1->getPrevPrevValue (),
//                                    a1,
//                                    a2,
//                                    a3,
//                                    a4,
//                                    a5);
//   valField1->setCurValue (val);
// #else
//   ALWAYS_ASSERT (0);
// #endif
// }

// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// template <bool useMetamaterials>
// ICUDA_DEVICE ICUDA_HOST
// void
// INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPML (time_step t,
//                                                                    TC pos,
//                                                                    IGRID<TC> *grid,
//                                                                    IGRID<TC> *gridPML1,
//                                                                    IGRID<TC> *gridPML2,
//                                                                    GridType gridType,
//                                                                    GridType gridPMLType1,
//                                                                    IGRID<TC> *materialGrid1,
//                                                                    GridType materialGridType1,
//                                                                    IGRID<TC> *materialGrid4,
//                                                                    GridType materialGridType4,
//                                                                    IGRID<TC> *materialGrid5,
//                                                                    GridType materialGridType5,
//                                                                    FPValue materialModifier)
// {
//   FPValue eps0 = PhysicsConst::Eps0;
//
//   TC posAbs = gridPML2->getTotalPosition (pos);
//
//   FieldPointValue *valField = gridPML2->getFieldPointValue (pos);
//
//   FieldPointValue *valField1;
//
//   if (useMetamaterials)
//   {
//     valField1 = gridPML1->getFieldPointValue (pos);
//   }
//   else
//   {
//     valField1 = grid->getFieldPointValue (pos);
//   }
//
//   FPValue material1 = materialGrid1 ? getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1) : 0;
//   FPValue material4 = materialGrid4 ? getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4) : 0;
//   FPValue material5 = materialGrid5 ? getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5) : 0;
//
//   FPValue modifier = material1 * materialModifier;
//   if (useMetamaterials)
//   {
//     modifier = 1;
//   }
//
//   FPValue k_mod1 = 1;
//   FPValue k_mod2 = 1;
//
//   FPValue Ca = (2 * eps0 * k_mod2 - material5 * gridTimeStep) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
//   FPValue Cb = ((2 * eps0 * k_mod1 + material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
//   FPValue Cc = ((2 * eps0 * k_mod1 - material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
//
// #if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
//   FieldValue val = calcFieldFromDOrB (valField->getPrevValue (),
//                                       valField1->getCurValue (),
//                                       valField1->getPrevValue (),
//                                       Ca,
//                                       Cb,
//                                       Cc);
// #else
//   ALWAYS_ASSERT (0);
// #endif
//
//   valField->setCurValue (val);
// }

// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// template <uint8_t grid_type>
// ICUDA_DEVICE ICUDA_HOST
// void
// INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationBorder (time_step t,
//                                                                       TC pos,
//                                                                       IGRID<TC> *grid,
//                                                                       SourceCallBack borderFunc)
// {
//   TC posAbs = grid->getTotalPosition (pos);
//
//   if (doSkipBorderFunc (posAbs, grid))
//   {
//     return;
//   }
//
//   TCFP realCoord;
//   FPValue timestep;
//   switch (grid_type)
//   {
//     case (static_cast<uint8_t> (GridType::EX)):
//     {
//       realCoord = yeeLayout->getExCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::EY)):
//     {
//       realCoord = yeeLayout->getEyCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::EZ)):
//     {
//       realCoord = yeeLayout->getEzCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HX)):
//     {
//       realCoord = yeeLayout->getHxCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HY)):
//     {
//       realCoord = yeeLayout->getHyCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HZ)):
//     {
//       realCoord = yeeLayout->getHzCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     default:
//     {
//       UNREACHABLE;
//     }
//   }
//
//   grid->getFieldPointValue (pos)->setCurValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep));
// }

// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// template <uint8_t grid_type>
// ICUDA_DEVICE ICUDA_HOST
// void
// INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationExact (time_step t,
//                                                                      TC pos,
//                                                                      IGRID<TC> *grid,
//                                                                      SourceCallBack exactFunc,
//                                                                      FPValue &normRe,
//                                                                      FPValue &normIm,
//                                                                      FPValue &normMod,
//                                                                      FPValue &maxRe,
//                                                                      FPValue &maxIm,
//                                                                      FPValue &maxMod)
// {
//   TC posAbs = grid->getTotalPosition (pos);
//
//   TCFP realCoord;
//   FPValue timestep;
//   switch (grid_type)
//   {
//     case (static_cast<uint8_t> (GridType::EX)):
//     {
//       realCoord = yeeLayout->getExCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::EY)):
//     {
//       realCoord = yeeLayout->getEyCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::EZ)):
//     {
//       realCoord = yeeLayout->getEzCoordFP (posAbs);
//       timestep = t + 0.5;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HX)):
//     {
//       realCoord = yeeLayout->getHxCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HY)):
//     {
//       realCoord = yeeLayout->getHyCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     case (static_cast<uint8_t> (GridType::HZ)):
//     {
//       realCoord = yeeLayout->getHzCoordFP (posAbs);
//       timestep = t + 1.0;
//       break;
//     }
//     default:
//     {
//       UNREACHABLE;
//     }
//   }
//
//   FieldValue numerical = grid->getFieldPointValue (pos)->getCurValue ();
//   FieldValue exact = exactFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);
//
// #ifdef COMPLEX_FIELD_VALUES
//   FPValue modExact = sqrt (SQR (exact.real ()) + SQR (exact.imag ()));
//   FPValue modNumerical = sqrt (SQR (numerical.real ()) + SQR (numerical.imag ()));
//
//   //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact.real (), numerical.real ());
//
//   normRe += SQR (exact.real () - numerical.real ());
//   normIm += SQR (exact.imag () - numerical.imag ());
//   normMod += SQR (modExact - modNumerical);
//
//   FPValue exactAbs = fabs (exact.real ());
//   if (maxRe < exactAbs)
//   {
//     maxRe = exactAbs;
//   }
//
//   exactAbs = fabs (exact.imag ());
//   if (maxIm < exactAbs)
//   {
//     maxIm = exactAbs;
//   }
//
//   exactAbs = modExact;
//   if (maxMod < exactAbs)
//   {
//     maxMod = exactAbs;
//   }
// #else
//   normRe += SQR (exact - numerical);
//
//   //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact, numerical);
//
//   FPValue exactAbs = fabs (exact);
//   if (maxRe < exactAbs)
//   {
//     maxRe = exactAbs;
//   }
// #endif
// }

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

#ifndef GPU_INTERNAL_SCHEME

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
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos42, 0));
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
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0));
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
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0));
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
                                                      omega, gamma);
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
                                                      omega, gamma);
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
                                                      omega, gamma);
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

#endif /* !GPU_INTERNAL_SCHEME */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::INTERNAL_SCHEME_BASE ()
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
  , CaEx (NULLPTR)
  , CbEx (NULLPTR)
  , CaEy (NULLPTR)
  , CbEy (NULLPTR)
  , CaEz (NULLPTR)
  , CbEz (NULLPTR)
  , DaHx (NULLPTR)
  , DbHx (NULLPTR)
  , DaHy (NULLPTR)
  , DbHy (NULLPTR)
  , DaHz (NULLPTR)
  , DbHz (NULLPTR)
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
#ifndef GPU_INTERNAL_SCHEME
  , totalTimeSteps (0)
  , NTimeSteps (0)
  , useParallel (false)
  , gpuIntScheme (NULLPTR)
  , gpuIntSchemeOnGPU (NULLPTR)
  , d_gpuIntSchemeOnGPU (NULLPTR)
#endif /* !GPU_INTERNAL_SCHEME */
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

#ifndef GPU_INTERNAL_SCHEME

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::initBlocks (time_step t_total)
{
  totalTimeSteps = t_total;

  /*
   * TODO: currently only single block is set up here, but underlying methods should support more?
   */
  blockCount = TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

  // TODO: allocate previous step storage for cuda blocks (see page 81)

#ifdef PARALLEL_GRID
  ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;
  blockSize = parallelYeeLayout->getSizeForCurNode ();
#else
  blockSize = yeeLayout->getSize ();
#endif

#ifdef PARALLEL_GRID
  if (useParallel)
  {
    time_step parallelBuf = (time_step) SOLVER_SETTINGS.getBufferSize ();
    NTimeSteps = parallelBuf;
  }
  else
#endif /* PARALLEL_GRID */
  {
    NTimeSteps = totalTimeSteps;
  }

#ifdef CUDA_ENABLED
  if (blockCount.calculateTotalCoord () > 1)
  {
    /*
     * More than one block is used, have to consider buffers now
     */
    time_step cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();

#ifdef PARALLEL_GRID
    if (useParallel)
    {
      /*
       * Cuda grid buffer can't be greater than parallel grid buffer, because there will be no data to fill it with.
       * If cuda grid buffer is less than parallel grid buffer, then parallel grid buffer won't be used fully, which
       * is undesirable. So, restrict buffers to be identical for the case of both parallel mode and cuda mode.
       */
      ALWAYS_ASSERT (cudaBuf == (time_step) SOLVER_SETTINGS.getBufferSize ())
    }
#endif /* PARALLEL_GRID */

    NTimeSteps = cudaBuf;
  }

  /*
   * Init InternalScheme on GPU
   */
  time_step cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();

  gpuIntScheme = new InternalSchemeGPU<Type, TCoord, layout_type> ();
  gpuIntSchemeOnGPU = new InternalSchemeGPU<Type, TCoord, layout_type> ();

  gpuIntScheme->initFromCPU (this, blockSize, TC_COORD (cudaBuf, cudaBuf, cudaBuf, ct1, ct2, ct3));
  gpuIntSchemeOnGPU->initOnGPU (gpuIntScheme);

  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>)));
#endif /* CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performNStepsForBlock (time_step tStart, time_step N, TC blockIdx)
{
#ifdef CUDA_ENABLED
  /*
   * Copy InternalScheme to GPU
   */
  gpuIntScheme->copyFromCPU (blockIdx * blockSize, blockSize);
  gpuIntSchemeOnGPU->copyToGPU (gpuIntScheme);
  cudaCheckErrorCmd (cudaMemcpy (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));
#endif /* CUDA_ENABLED */

  TC zero = TC_COORD (0, 0, 0, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
  GridCoordinate3D zero3D = GRID_COORDINATE_3D (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D ExSize = gpuIntScheme->doNeedEx ? expandTo3D (gpuIntScheme->getEx ()->getSize (), ct1, ct2, ct3) : zero3D;
  GridCoordinate3D EySize = gpuIntScheme->doNeedEy ? expandTo3D (gpuIntScheme->getEy ()->getSize (), ct1, ct2, ct3) : zero3D;
  GridCoordinate3D EzSize = gpuIntScheme->doNeedEz ? expandTo3D (gpuIntScheme->getEz ()->getSize (), ct1, ct2, ct3) : zero3D;
  GridCoordinate3D HxSize = gpuIntScheme->doNeedHx ? expandTo3D (gpuIntScheme->getHx ()->getSize (), ct1, ct2, ct3) : zero3D;
  GridCoordinate3D HySize = gpuIntScheme->doNeedHy ? expandTo3D (gpuIntScheme->getHy ()->getSize (), ct1, ct2, ct3) : zero3D;
  GridCoordinate3D HzSize = gpuIntScheme->doNeedHz ? expandTo3D (gpuIntScheme->getHz ()->getSize (), ct1, ct2, ct3) : zero3D;
#endif

  for (time_step t = tStart; t < tStart + N; ++t)
  {
    DPRINTF (LOG_LEVEL_NONE, "calculating time step %d\n", t);

#ifdef CUDA_ENABLED
    TC ExStart = gpuIntScheme->doNeedEx ? gpuIntScheme->Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = gpuIntScheme->doNeedEx ? gpuIntScheme->Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = gpuIntScheme->doNeedEy ? gpuIntScheme->Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = gpuIntScheme->doNeedEy ? gpuIntScheme->Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = gpuIntScheme->doNeedEz ? gpuIntScheme->Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = gpuIntScheme->doNeedEz ? gpuIntScheme->Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = gpuIntScheme->doNeedHx ? gpuIntScheme->Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = gpuIntScheme->doNeedHx ? gpuIntScheme->Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = gpuIntScheme->doNeedHy ? gpuIntScheme->Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = gpuIntScheme->doNeedHy ? gpuIntScheme->Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = gpuIntScheme->doNeedHz ? gpuIntScheme->Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = gpuIntScheme->doNeedHz ? gpuIntScheme->Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
#else /* CUDA_ENABLED */
    TC ExStart = doNeedEx ? Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = doNeedEx ? Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = doNeedEy ? Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = doNeedEy ? Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = doNeedEz ? Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = doNeedEz ? Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = doNeedHx ? Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = doNeedHx ? Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = doNeedHy ? Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = doNeedHy ? Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = doNeedHz ? Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = doNeedHz ? Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
#endif /* CUDA_ENABLED */

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->performPlaneWaveEStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getEInc ()->getSize ());
      gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchEInc (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEInc ()->nextTimeStep ();
#else /* CUDA_ENABLED */
      performPlaneWaveESteps (t, zero1D, getEInc ()->getSize ());
      getEInc ()->shiftInTime ();
      getEInc ()->nextTimeStep ();
#endif /* !CUDA_ENABLED */
    }

    if (getDoNeedEx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEx (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1x (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1x ()->nextTimeStep ();
      }
#else
      getEx ()->shiftInTime ();
      getEx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getDx ()->shiftInTime ();
        getDx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getD1x ()->shiftInTime ();
        getD1x ()->nextTimeStep ();
      }
#endif
    }

    if (getDoNeedEy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEy (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1y (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1y ()->nextTimeStep ();
      }
#else
      getEy ()->shiftInTime ();
      getEy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getDy ()->shiftInTime ();
        getDy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getD1y ()->shiftInTime ();
        getD1y ()->nextTimeStep ();
      }
#endif
    }

    if (getDoNeedEz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEz (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1z (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1z ()->nextTimeStep ();
      }
#else
      getEz ()->shiftInTime ();
      getEz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getDz ()->shiftInTime ();
        getDz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getD1z ()->shiftInTime ();
        getD1z ()->nextTimeStep ();
      }
#endif
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->performPlaneWaveHStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getHInc ()->getSize ());
      gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchHInc (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHInc ()->nextTimeStep ();
#else /* CUDA_ENABLED */
      performPlaneWaveHSteps (t, zero1D, getHInc ()->getSize ());
      getHInc ()->shiftInTime ();
      getHInc ()->nextTimeStep ();
#endif /* !CUDA_ENABLED */
    }

    if (getDoNeedHx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHx (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1x (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1x ()->nextTimeStep ();
      }
#else
      getHx ()->shiftInTime ();
      getHx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getBx ()->shiftInTime ();
        getBx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getB1x ()->shiftInTime ();
        getB1x ()->nextTimeStep ();
      }
#endif
    }

    if (getDoNeedHy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHy (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1y (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1y ()->nextTimeStep ();
      }
#else
      getHy ()->shiftInTime ();
      getHy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getBy ()->shiftInTime ();
        getBy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getB1y ()->shiftInTime ();
        getB1y ()->nextTimeStep ();
      }
#endif
    }

    if (getDoNeedHz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHz (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1z (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1z ()->nextTimeStep ();
      }
#else
      getHz ()->shiftInTime ();
      getHz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        getBz ()->shiftInTime ();
        getBz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        getB1z ()->shiftInTime ();
        getB1z ()->nextTimeStep ();
      }
#endif
    }
  }

#ifdef CUDA_ENABLED
  /*
   * Copy back from GPU to CPU
   */
  bool finalCopy = blockIdx + TC_COORD (1, 1, 1, ct1, ct2, ct3) == blockCount;
  gpuIntScheme->copyBackToCPU (NTimeSteps, finalCopy);
#endif /* CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::share ()
{
#ifdef PARALLEL_GRID
  if (!useParallel)
  {
    return;
  }

  if (intScheme->getDoNeedEx ())
  {
    ASSERT (((ParallelGrid *) Ex)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Ex)->share ();
    ((ParallelGrid *) Ex)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) Dx)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) Dx)->share ();
      ((ParallelGrid *) Dx)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) D1x)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) D1x)->share ();
      ((ParallelGrid *) D1x)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedEy ())
  {
    ASSERT (((ParallelGrid *) Ey)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Ey)->share ();
    ((ParallelGrid *) Ey)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) Dy)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) Dy)->share ();
      ((ParallelGrid *) Dy)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) D1y)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) D1y)->share ();
      ((ParallelGrid *) D1y)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedEz ())
  {
    ASSERT (((ParallelGrid *) Ez)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Ez)->share ();
    ((ParallelGrid *) Ez)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) Dz)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) Dz)->share ();
      ((ParallelGrid *) Dz)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) D1z)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) D1z)->share ();
      ((ParallelGrid *) D1z)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHx ())
  {
    ASSERT (((ParallelGrid *) Hx)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Hx)->share ();
    ((ParallelGrid *) Hx)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) Bx)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) Bx)->share ();
      ((ParallelGrid *) Bx)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) B1x)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) B1x)->share ();
      ((ParallelGrid *) B1x)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHy ())
  {
    ASSERT (((ParallelGrid *) Hy)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Hy)->share ();
    ((ParallelGrid *) Hy)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) By)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) By)->share ();
      ((ParallelGrid *) By)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) B1y)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) B1y)->share ();
      ((ParallelGrid *) B1y)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHz ())
  {
    ASSERT (((ParallelGrid *) Hz)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) Hz)->share ();
    ((ParallelGrid *) Hz)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) Bz)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) Bz)->share ();
      ((ParallelGrid *) Bz)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) B1z)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) B1z)->share ();
      ((ParallelGrid *) B1z)->zeroShareStep ();
    }
  }
#endif /* PARALLEL_GRID */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::rebalance ()
{

}

/**
 * Perform computations of single time step for specific field and for specified chunk.
 *
 * NOTE: For GPU InternalScheme this method is not defined, because it is supposed to be ran on CPU only,
 *       and call kernels deeper in call tree.
 *
 * NOTE: Start and End coordinates should correctly consider buffers in parallel grid,
 *       which means, that computations are not performed for incorrect grid points.
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStep (time_step t, /**< time step to calculate */
                                                                     TC start, /**< start coordinate of chunk to perform computations on */
                                                                     TC end) /**< end coordinate of chunk to perform computations on */
{
  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

  IGRID<TC> *grid = NULLPTR;
  GridType gridType = GridType::NONE;

  IGRID<TC> *materialGrid = NULLPTR;
  GridType materialGridType = GridType::NONE;

  IGRID<TC> *materialGrid1 = NULLPTR;
  GridType materialGridType1 = GridType::NONE;

  IGRID<TC> *materialGrid2 = NULLPTR;
  GridType materialGridType2 = GridType::NONE;

  IGRID<TC> *materialGrid3 = NULLPTR;
  GridType materialGridType3 = GridType::NONE;

  IGRID<TC> *materialGrid4 = NULLPTR;
  GridType materialGridType4 = GridType::NONE;

  IGRID<TC> *materialGrid5 = NULLPTR;
  GridType materialGridType5 = GridType::NONE;

  IGRID<TC> *oppositeGrid1 = NULLPTR;
  IGRID<TC> *oppositeGrid2 = NULLPTR;

  IGRID<TC> *gridPML1 = NULLPTR;
  GridType gridPMLType1 = GridType::NONE;

  IGRID<TC> *gridPML2 = NULLPTR;
  GridType gridPMLType2 = GridType::NONE;

  IGRID<TC> *Ca = NULLPTR;
  IGRID<TC> *Cb = NULLPTR;

  SourceCallBack rightSideFunc = NULLPTR;
  SourceCallBack borderFunc = NULLPTR;
  SourceCallBack exactFunc = NULLPTR;

  TCS diff11;
  TCS diff12;
  TCS diff21;
  TCS diff22;

  /*
   * TODO: remove this, multiply on this at initialization
   */
  FPValue materialModifier;

  calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&grid, &gridType,
    &materialGrid, &materialGridType, &materialGrid1, &materialGridType1, &materialGrid2, &materialGridType2,
    &materialGrid3, &materialGridType3, &materialGrid4, &materialGridType4, &materialGrid5, &materialGridType5,
    &oppositeGrid1, &oppositeGrid2, &gridPML1, &gridPMLType1, &gridPML2, &gridPMLType2,
    &rightSideFunc, &borderFunc, &exactFunc, &materialModifier, &Ca, &Cb);

  calculateFieldStepInitDiff<grid_type> (&diff11, &diff12, &diff21, &diff22);

#ifdef CUDA_ENABLED
  CudaGrid<TC> *d_grid = NULLPTR;
  GridType _gridType = GridType::NONE;

  CudaGrid<TC> *d_materialGrid = NULLPTR;
  GridType _materialGridType = GridType::NONE;

  CudaGrid<TC> *d_materialGrid1 = NULLPTR;
  GridType _materialGridType1 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid2 = NULLPTR;
  GridType _materialGridType2 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid3 = NULLPTR;
  GridType _materialGridType3 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid4 = NULLPTR;
  GridType _materialGridType4 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid5 = NULLPTR;
  GridType _materialGridType5 = GridType::NONE;

  CudaGrid<TC> *d_oppositeGrid1 = NULLPTR;
  CudaGrid<TC> *d_oppositeGrid2 = NULLPTR;

  CudaGrid<TC> *d_gridPML1 = NULLPTR;
  GridType _gridPMLType1 = GridType::NONE;

  CudaGrid<TC> *d_gridPML2 = NULLPTR;
  GridType _gridPMLType2 = GridType::NONE;

  CudaGrid<TC> *d_Ca = NULLPTR;
  CudaGrid<TC> *d_Cb = NULLPTR;

  SourceCallBack _rightSideFunc = NULLPTR;
  SourceCallBack _borderFunc = NULLPTR;
  SourceCallBack _exactFunc = NULLPTR;

  FPValue _materialModifier;

  TCS _diff11;
  TCS _diff12;
  TCS _diff21;
  TCS _diff22;

  gpuIntSchemeOnGPU->template calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&d_grid, &_gridType,
    &d_materialGrid, &_materialGridType, &d_materialGrid1, &_materialGridType1, &d_materialGrid2, &_materialGridType2,
    &d_materialGrid3, &_materialGridType3, &d_materialGrid4, &_materialGridType4, &d_materialGrid5, &_materialGridType5,
    &d_oppositeGrid1, &d_oppositeGrid2, &d_gridPML1, &_gridPMLType1, &d_gridPML2, &_gridPMLType2,
    &_rightSideFunc, &_borderFunc, &_exactFunc, &_materialModifier, &d_Ca, &d_Cb);

  gpuIntScheme->template calculateFieldStepInitDiff<grid_type> (&_diff11, &_diff12, &_diff21, &_diff22);

#endif /* CUDA_ENABLED */

  // TODO: specialize for each dimension
  GridCoordinate3D start3D;
  GridCoordinate3D end3D;

  expandTo3DStartEnd (start, end, start3D, end3D, ct1, ct2, ct3);

  // TODO: remove this check for each iteration
  if (t > 0)
  {
#ifdef CUDA_ENABLED

    // Launch kernel here
    gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <grid_type, usePML, useMetamaterials> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                            t, diff11, diff12, diff21, diff22,
                                                                            d_grid,
                                                                            d_oppositeGrid1, d_oppositeGrid2, _rightSideFunc, d_Ca, d_Cb);

#else /* CUDA_ENABLED */

    for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
      {
        // TODO: check that this is optimized out in case 2D mode
        for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);

          // TODO: add getTotalPositionDiff here, which will be called before loop
          TC posAbs = grid->getTotalPosition (pos);

          TCFP coordFP;

          if (rightSideFunc != NULLPTR)
          {
            switch (grid_type)
            {
              case (static_cast<uint8_t> (GridType::EX)):
              {
                coordFP = yeeLayout->getExCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::EY)):
              {
                coordFP = yeeLayout->getEyCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::EZ)):
              {
                coordFP = yeeLayout->getEzCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HX)):
              {
                coordFP = yeeLayout->getHxCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HY)):
              {
                coordFP = yeeLayout->getHyCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HZ)):
              {
                coordFP = yeeLayout->getHzCoordFP (posAbs);
                break;
              }
              default:
              {
                UNREACHABLE;
              }
            }
          }

          calculateFieldStepIteration<grid_type, usePML> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                          grid, coordFP,
                                                          oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb);
        }
      }
    }
#endif

//     if (usePML)
//     {
//       if (useMetamaterials)
//       {
// #ifdef TWO_TIME_STEPS
//         for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
//         {
//           // TODO: check that this loop is optimized out
//           for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
//           {
//             // TODO: check that this loop is optimized out
//             for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
//             {
//               TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
//               calculateFieldStepIterationPMLMetamaterials (t, pos, grid, gridPML1, gridType,
//                 materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
//                 materialModifier);
//             }
//           }
//         }
// #else
//         ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
// #endif
//       }
//
//       for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
//       {
//         // TODO: check that this loop is optimized out
//         for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
//         {
//           // TODO: check that this loop is optimized out
//           for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
//           {
//             TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
//             calculateFieldStepIterationPML<useMetamaterials> (t, pos, grid, gridPML1, gridPML2, gridType, gridPMLType1,
//               materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
//               materialModifier);
//           }
//         }
//       }
//     }
  }

  // if (borderFunc != NULLPTR)
  // {
  //   GridCoordinate3D startBorder;
  //   GridCoordinate3D endBorder;
  //
  //   expandTo3DStartEnd (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3),
  //                       grid->getSize (),
  //                       startBorder,
  //                       endBorder,
  //                       ct1, ct2, ct3);
  //
  //   for (grid_coord i = startBorder.get1 (); i < endBorder.get1 (); ++i)
  //   {
  //     // TODO: check that this loop is optimized out
  //     for (grid_coord j = startBorder.get2 (); j < endBorder.get2 (); ++j)
  //     {
  //       // TODO: check that this loop is optimized out
  //       for (grid_coord k = startBorder.get3 (); k < endBorder.get3 (); ++k)
  //       {
  //         TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
  //         calculateFieldStepIterationBorder<grid_type> (t, pos, grid, borderFunc);
  //       }
  //     }
  //   }
  // }

//   if (exactFunc != NULLPTR)
//   {
//     FPValue normRe = 0.0;
//     FPValue normIm = 0.0;
//     FPValue normMod = 0.0;
//
//     FPValue maxRe = 0.0;
//     FPValue maxIm = 0.0;
//     FPValue maxMod = 0.0;
//
//     GridCoordinate3D startNorm = start3D;
//     GridCoordinate3D endNorm = end3D;
//
//     if (SOLVER_SETTINGS.getExactSolutionCompareStartX () != 0)
//     {
//       startNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareStartX ());
//     }
//     if (SOLVER_SETTINGS.getExactSolutionCompareStartY () != 0)
//     {
//       startNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareStartY ());
//     }
//     if (SOLVER_SETTINGS.getExactSolutionCompareStartZ () != 0)
//     {
//       startNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareStartZ ());
//     }
//
//     if (SOLVER_SETTINGS.getExactSolutionCompareEndX () != 0)
//     {
//       endNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareEndX ());
//     }
//     if (SOLVER_SETTINGS.getExactSolutionCompareEndY () != 0)
//     {
//       endNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareEndY ());
//     }
//     if (SOLVER_SETTINGS.getExactSolutionCompareEndZ () != 0)
//     {
//       endNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareEndZ ());
//     }
//
//     IGRID<TC> *normGrid = grid;
//     if (usePML)
//     {
//       grid = gridPML2;
//     }
//
//     for (grid_coord i = startNorm.get1 (); i < endNorm.get1 (); ++i)
//     {
//       // TODO: check that this loop is optimized out
//       for (grid_coord j = startNorm.get2 (); j < endNorm.get2 (); ++j)
//       {
//         // TODO: check that this loop is optimized out
//         for (grid_coord k = startNorm.get3 (); k < endNorm.get3 (); ++k)
//         {
//           TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
//           calculateFieldStepIterationExact<grid_type> (t, pos, grid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
//         }
//       }
//     }
//
// #ifdef COMPLEX_FIELD_VALUES
//     normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
//     normIm = sqrt (normIm / grid->getSize ().calculateTotalCoord ());
//     normMod = sqrt (normMod / grid->getSize ().calculateTotalCoord ());
//
//     /*
//      * NOTE: do not change this! test suite depdends on the order of values in output
//      */
//     printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
//       grid->getName (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
// #else
//     normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
//
//     /*
//      * NOTE: do not change this! test suite depdends on the order of values in output
//      */
//     printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
//       grid->getName (), t, normRe, normRe * 100.0 / maxRe);
// #endif
//   }
}

#endif /* !GPU_INTERNAL_SCHEME */

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
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
ICUDA_DEVICE ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStep (time_step t, TC start, TC end)
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
ICUDA_DEVICE ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepInit (IGRID<TC> **grid, GridType *gridType, IGRID<TC> **materialGrid, GridType *materialGridType, IGRID<TC> **materialGrid1, GridType *materialGridType1,
IGRID<TC> **materialGrid2, GridType *materialGridType2, IGRID<TC> **materialGrid3, GridType *materialGridType3, IGRID<TC> **materialGrid4, GridType *materialGridType4,
IGRID<TC> **materialGrid5, GridType *materialGridType5, IGRID<TC> **oppositeGrid1, IGRID<TC> **oppositeGrid2, IGRID<TC> **gridPML1, GridType *gridPMLType1, IGRID<TC> **gridPML2, GridType *gridPMLType2,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML>
ICUDA_DEVICE ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIteration (time_step t,
                                                               TC pos,
                                                               IGRID<TC> *grid,
                                                               GridType gridType,
                                                               IGRID<TC> *materialGrid,
                                                               GridType materialGridType,
                                                               IGRID<TC> *oppositeGrid1,
                                                               IGRID<TC> *oppositeGrid2,
                                                               SourceCallBack rightSideFunc,
                                                               FPValue materialModifier)
{
  FPValue eps0 = PhysicsConst::Eps0;

  // TODO: add getTotalPositionDiff here, which will be called before loop
  TC posAbs = grid->getTotalPosition (pos);
  // TODO: [possible] move 1D gridValues to 3D gridValues array
  FieldPointValue *valField = grid->getFieldPointValue (pos);

  FPValue material = materialGrid ? getMaterial (posAbs, gridType, materialGrid, materialGridType) : 0;

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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPMLMetamaterials (time_step t,
                                                                               TC pos,
                                                                               IGRID<TC> *grid,
                                                                               IGRID<TC> *gridPML,
                                                                               GridType gridType,
                                                                               IGRID<TC> *materialGrid1,
                                                                               GridType materialGridType1,
                                                                               IGRID<TC> *materialGrid2,
                                                                               GridType materialGridType2,
                                                                               IGRID<TC> *materialGrid3,
                                                                               GridType materialGridType3,
                                                                               FPValue materialModifier)
{
  TC posAbs = grid->getTotalPosition (pos);
  FieldPointValue *valField = grid->getFieldPointValue (pos);
  FieldPointValue *valField1 = gridPML->getFieldPointValue (pos);

  FPValue material1;
  FPValue material2;

  FPValue material = getMetaMaterial (posAbs, gridType,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool useMetamaterials>
ICUDA_DEVICE ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPML (time_step t,
                                                                   TC pos,
                                                                   IGRID<TC> *grid,
                                                                   IGRID<TC> *gridPML1,
                                                                   IGRID<TC> *gridPML2,
                                                                   GridType gridType,
                                                                   GridType gridPMLType1,
                                                                   IGRID<TC> *materialGrid1,
                                                                   GridType materialGridType1,
                                                                   IGRID<TC> *materialGrid4,
                                                                   GridType materialGridType4,
                                                                   IGRID<TC> *materialGrid5,
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE ICUDA_HOST
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

  grid->getFieldPointValue (pos)->setCurValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep));
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE ICUDA_HOST
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
        return yeeLayout->getApproximateMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos12),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos21),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos22),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos31),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos32),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos41),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos42));
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
        return yeeLayout->getApproximateMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos12));
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return yeeLayout->getApproximateMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos12),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos21),
                                                  gridMaterial->getFieldPointValueByAbsolutePos (absPos22));
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
        return yeeLayout->getApproximateMetaMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos22),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos31),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos32),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos41),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos42),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos22),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos31),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos32),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos41),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos42),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos22),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos31),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos32),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos41),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos42),
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
        return yeeLayout->getApproximateMetaMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos12),
                                                      omega, gamma);
      }
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      {
        return yeeLayout->getApproximateMetaMaterial (gridMaterial->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterial->getFieldPointValueByAbsolutePos (absPos22),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterialOmega->getFieldPointValueByAbsolutePos (absPos22),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos11),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos12),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos21),
                                                      gridMaterialGamma->getFieldPointValueByAbsolutePos (absPos22),
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

  if (setSource)
  {
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

    //printf ("EInc[0] %f \n", valE->getCurValue ());
  }

#ifdef ENABLE_ASSERTS
  GridCoordinate1D posEnd (EInc->getSize ().get1 () - 1, CoordinateType::X);
  ALWAYS_ASSERT (EInc->getFieldPointValue (posEnd)->getCurValue () == getFieldValueRealOnly (0.0));
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
  GridCoordinate1D pos (HInc->getSize ().get1 () - 2, CoordinateType::X);
  ALWAYS_ASSERT (HInc->getFieldPointValue (pos)->getCurValue () == getFieldValueRealOnly (0.0));
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

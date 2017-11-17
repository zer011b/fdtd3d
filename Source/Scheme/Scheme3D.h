#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "Scheme.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"

#ifdef GRID_3D

class Scheme3D: public Scheme
{
  YeeGridLayout *yeeLayout;

  Grid<GridCoordinate3D> *Ex;
  Grid<GridCoordinate3D> *Ey;
  Grid<GridCoordinate3D> *Ez;
  Grid<GridCoordinate3D> *Hx;
  Grid<GridCoordinate3D> *Hy;
  Grid<GridCoordinate3D> *Hz;

  Grid<GridCoordinate3D> *Dx;
  Grid<GridCoordinate3D> *Dy;
  Grid<GridCoordinate3D> *Dz;
  Grid<GridCoordinate3D> *Bx;
  Grid<GridCoordinate3D> *By;
  Grid<GridCoordinate3D> *Bz;

  Grid<GridCoordinate3D> *D1x;
  Grid<GridCoordinate3D> *D1y;
  Grid<GridCoordinate3D> *D1z;
  Grid<GridCoordinate3D> *B1x;
  Grid<GridCoordinate3D> *B1y;
  Grid<GridCoordinate3D> *B1z;

  Grid<GridCoordinate3D> *ExAmplitude;
  Grid<GridCoordinate3D> *EyAmplitude;
  Grid<GridCoordinate3D> *EzAmplitude;
  Grid<GridCoordinate3D> *HxAmplitude;
  Grid<GridCoordinate3D> *HyAmplitude;
  Grid<GridCoordinate3D> *HzAmplitude;

  Grid<GridCoordinate3D> *Eps;
  Grid<GridCoordinate3D> *Mu;

  Grid<GridCoordinate3D> *SigmaX;
  Grid<GridCoordinate3D> *SigmaY;
  Grid<GridCoordinate3D> *SigmaZ;

  Grid<GridCoordinate3D> *OmegaPE;
  Grid<GridCoordinate3D> *GammaE;

  Grid<GridCoordinate3D> *OmegaPM;
  Grid<GridCoordinate3D> *GammaM;

  Grid<GridCoordinate1D> *EInc;
  Grid<GridCoordinate1D> *HInc;

  Grid<GridCoordinate3D> *totalEx;
  Grid<GridCoordinate3D> *totalEy;
  Grid<GridCoordinate3D> *totalEz;
  Grid<GridCoordinate3D> *totalHx;
  Grid<GridCoordinate3D> *totalHy;
  Grid<GridCoordinate3D> *totalHz;

  bool totalInitialized;

  Grid<GridCoordinate3D> *totalEps;
  Grid<GridCoordinate3D> *totalMu;
  Grid<GridCoordinate3D> *totalOmegaPE;
  Grid<GridCoordinate3D> *totalOmegaPM;
  Grid<GridCoordinate3D> *totalGammaE;
  Grid<GridCoordinate3D> *totalGammaM;

  // Wave parameters
  FPValue sourceWaveLength;
  FPValue sourceFrequency;
  FPValue relPhaseVelocity;

  /** Courant number */
  FPValue courantNum;

  // dx
  FPValue gridStep;

  // dt
  FPValue gridTimeStep;

  time_step totalStep;

  int process;

  int numProcs;

  GridCoordinate3D leftNTFF;
  GridCoordinate3D rightNTFF;

  Dumper<GridCoordinate3D> *dumper[FILE_TYPE_COUNT];
  Loader<GridCoordinate3D> *loader[FILE_TYPE_COUNT];

  Dumper<GridCoordinate1D> *dumper1D[FILE_TYPE_COUNT];

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

private:

  template <uint8_t grid_coord, bool usePML, bool useMetamaterials>
  void calculateFieldStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateExStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  FieldValue approximateIncidentWave (GridCoordinateFP3D, FPValue, Grid<GridCoordinate1D> &);
  FieldValue approximateIncidentWaveE (GridCoordinateFP3D);
  FieldValue approximateIncidentWaveH (GridCoordinateFP3D);

  // template <uint8_t grid_type>
  // void calculateTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
  //                     GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);

  void calculateExTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateEyTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateEzTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHxTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHyTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHzTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);

  void performExSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHzSteps (time_step, GridCoordinate3D, GridCoordinate3D);

  template<uint8_t EnumVal> void performPointSourceCalc (time_step);

  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

  void makeGridScattered (Grid<GridCoordinate3D> *, GridType);
  void gatherFieldsTotal (bool);
  void saveGrids (time_step);
  void saveNTFF (bool, time_step);

  void additionalUpdateOfGrids (time_step, time_step &);

  GridCoordinate3D getStartCoord (GridType, GridCoordinate3D);
  GridCoordinate3D getEndCoord (GridType, GridCoordinate3D);

public:

  virtual void performSteps () CXX11_OVERRIDE;

  void initScheme (FPValue, FPValue);
  void initCallBacks ();
  void initGrids ();

  Scheme3D (YeeGridLayout *layout,
            const GridCoordinate3D& totSize,
            time_step tStep);

  ~Scheme3D ();

  struct NPair
  {
    FieldValue nTeta;
    FieldValue nPhi;

    NPair (FieldValue n_teta, FieldValue n_phi)
      : nTeta (n_teta)
    , nPhi (n_phi)
    {
    }

    NPair operator+ (const NPair &right)
    {
      return NPair (nTeta + right.nTeta, nPhi + right.nPhi);
    }
  };

  /*
   * 3D ntff
   */
  NPair ntffN_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffN_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffN_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  NPair ntffL_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffL_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffL_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  NPair ntffN (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
               Grid<GridCoordinate3D> *);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);

  FieldValue calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                        FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                        FPValue Ca, FPValue Cb, FPValue delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return Ca * prev + Cb * tmp;
  }

  FieldValue calcFieldDrude (FieldValue curDOrB, FieldValue prevDOrB, FieldValue prevPrevDOrB,
                             FieldValue prevEOrH, FieldValue prevPrevEOrH,
                             FPValue b0, FPValue b1, FPValue b2, FPValue a1, FPValue a2)
  {
    return b0 * curDOrB + b1 * prevDOrB + b2 * prevPrevDOrB - a1 * prevEOrH - a2 * prevPrevEOrH;
  }

  FieldValue calcFieldFromDOrB (FieldValue prevEOrH, FieldValue curDOrB, FieldValue prevDOrB,
                                FPValue Ca, FPValue Cb, FPValue Cc)
  {
    return Ca * prevEOrH + Cb * curDOrB - Cc * prevDOrB;
  }
};

template<uint8_t EnumVal>
void
Scheme3D::performPointSourceCalc (time_step t)
{
  Grid<GridCoordinate3D> *grid = NULLPTR;

  switch (EnumVal)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (grid);

  GridCoordinate3D pos (solverSettings.getPointSourcePositionX (),
                        solverSettings.getPointSourcePositionY (),
                        solverSettings.getPointSourcePositionZ ());

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
//
// template <uint8_t grid_type>
// void calculateTFSF (GridCoordinate3D posAbs,
//                     FieldValue &valOpposite11,
//                     FieldValue &valOpposite12,
//                     FieldValue &valOpposite21,
//                     FieldValue &valOpposite22,
//                     GridCoordinate3D pos11,
//                     GridCoordinate3D pos12,
//                     GridCoordinate3D pos21,
//                     GridCoordinate3D pos22)
// {
//   switch (grid_type)
//   {
//     case (static_cast<uint8_t> (GridType::EX)):
//     {
//       bool do_need_update_down = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN, DO_USE_3D_MODE);
//       bool do_need_update_up = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP, DO_USE_3D_MODE);
//
//       bool do_need_update_back = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK, DO_USE_3D_MODE);
//       bool do_need_update_front = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT, DO_USE_3D_MODE);
//
//   GridCoordinate3D auxPosY;
//   GridCoordinate3D auxPosZ;
//   FieldValue diffY;
//   FieldValue diffZ;
//
//   if (do_need_update_down)
//   {
//     auxPosY = posUp;
//   }
//   else if (do_need_update_up)
//   {
//     auxPosY = posDown;
//   }
//
//   if (do_need_update_back)
//   {
//     auxPosZ = posFront;
//   }
//   else if (do_need_update_front)
//   {
//     auxPosZ = posBack;
//   }
//
//   if (do_need_update_down || do_need_update_up)
//   {
//     GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPosY));
//
//     diffY = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
//   }
//
//   if (do_need_update_back || do_need_update_front)
//   {
//     GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPosZ));
//
//     diffZ = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
//   }
//
//   if (do_need_update_down)
//   {
//     valOpposite11 -= diffY;
//   }
//   else if (do_need_update_up)
//   {
//     valOpposite12 -= diffY;
//   }
//
//   if (do_need_update_back)
//   {
//     valOpposite21 -= diffZ;
//   }
//   else if (do_need_update_front)
//   {
//     valOpposite22 -= diffZ;
//   }
// }

template<uint8_t grid_type, bool usePML, bool useMetamaterials>
void
Scheme3D::calculateFieldStep (time_step t, GridCoordinate3D start, GridCoordinate3D end)
{
  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

  FPValue eps0 = PhysicsConst::Eps0;

  Grid<GridCoordinate3D> *grid = NULLPTR;
  GridType gridType = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid = NULLPTR;
  GridType materialGridType = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid1 = NULLPTR;
  GridType materialGridType1 = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid2 = NULLPTR;
  GridType materialGridType2 = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid3 = NULLPTR;
  GridType materialGridType3 = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid4 = NULLPTR;
  GridType materialGridType4 = GridType::NONE;

  Grid<GridCoordinate3D> *materialGrid5 = NULLPTR;
  GridType materialGridType5 = GridType::NONE;

  Grid<GridCoordinate3D> *oppositeGrid1 = NULLPTR;
  Grid<GridCoordinate3D> *oppositeGrid2 = NULLPTR;

  Grid<GridCoordinate3D> *gridPML1 = NULLPTR;
  GridType gridPMLType1 = GridType::NONE;

  Grid<GridCoordinate3D> *gridPML2 = NULLPTR;
  GridType gridPMLType2 = GridType::NONE;

  SourceCallBack rightSideFunc = NULLPTR;
  SourceCallBack borderFunc = NULLPTR;
  SourceCallBack exactFunc = NULLPTR;

  /*
   * TODO: remove this, multiply on this at initialization
   */
  FPValue materialModifier;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      gridType = GridType::EX;

      materialGrid = Eps;
      materialGridType = GridType::EPS;
      materialModifier = PhysicsConst::Eps0;

      oppositeGrid1 = Hz;
      oppositeGrid2 = Hy;

      rightSideFunc = Jx;
      borderFunc = ExBorder;
      exactFunc = ExExact;

      if (usePML)
      {
        grid = Dx;
        gridType = GridType::DX;

        gridPML1 = D1x;
        gridPMLType1 = GridType::DX;

        gridPML2 = Ex;
        gridPMLType2 = GridType::EX;

        materialGrid = SigmaY;
        materialGridType = GridType::SIGMAY;

        materialGrid1 = Eps;
        materialGridType1 = GridType::EPS;

        materialGrid4 = SigmaX;
        materialGridType4 = GridType::SIGMAX;

        materialGrid5 = SigmaZ;
        materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPE;
          materialGridType2 = GridType::OMEGAPE;

          materialGrid3 = GammaE;
          materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      gridType = GridType::EY;

      materialGrid = Eps;
      materialGridType = GridType::EPS;
      materialModifier = PhysicsConst::Eps0;

      oppositeGrid1 = Hx;
      oppositeGrid2 = Hz;

      rightSideFunc = Jy;
      borderFunc = EyBorder;
      exactFunc = EyExact;

      if (usePML)
      {
        grid = Dy;
        gridType = GridType::DY;

        gridPML1 = D1y;
        gridPMLType1 = GridType::DY;

        gridPML2 = Ey;
        gridPMLType2 = GridType::EY;

        materialGrid = SigmaZ;
        materialGridType = GridType::SIGMAZ;

        materialGrid1 = Eps;
        materialGridType1 = GridType::EPS;

        materialGrid4 = SigmaY;
        materialGridType4 = GridType::SIGMAY;

        materialGrid5 = SigmaX;
        materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPE;
          materialGridType2 = GridType::OMEGAPE;

          materialGrid3 = GammaE;
          materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      gridType = GridType::EZ;

      materialGrid = Eps;
      materialGridType = GridType::EPS;
      materialModifier = PhysicsConst::Eps0;

      oppositeGrid1 = Hy;
      oppositeGrid2 = Hx;

      rightSideFunc = Jz;
      borderFunc = EzBorder;
      exactFunc = EzExact;

      if (usePML)
      {
        grid = Dz;
        gridType = GridType::DZ;

        gridPML1 = D1z;
        gridPMLType1 = GridType::DZ;

        gridPML2 = Ez;
        gridPMLType2 = GridType::EZ;

        materialGrid = SigmaX;
        materialGridType = GridType::SIGMAX;

        materialGrid1 = Eps;
        materialGridType1 = GridType::EPS;

        materialGrid4 = SigmaZ;
        materialGridType4 = GridType::SIGMAZ;

        materialGrid5 = SigmaY;
        materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPE;
          materialGridType2 = GridType::OMEGAPE;

          materialGrid3 = GammaE;
          materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      gridType = GridType::HX;

      materialGrid = Mu;
      materialGridType = GridType::MU;
      materialModifier = PhysicsConst::Mu0;

      oppositeGrid1 = Ey;
      oppositeGrid2 = Ez;

      rightSideFunc = Mx;
      borderFunc = HxBorder;
      exactFunc = HxExact;

      if (usePML)
      {
        grid = Bx;
        gridType = GridType::BX;

        gridPML1 = B1x;
        gridPMLType1 = GridType::BX;

        gridPML2 = Hx;
        gridPMLType2 = GridType::HX;

        materialGrid = SigmaY;
        materialGridType = GridType::SIGMAY;

        materialGrid1 = Mu;
        materialGridType1 = GridType::MU;

        materialGrid4 = SigmaX;
        materialGridType4 = GridType::SIGMAX;

        materialGrid5 = SigmaZ;
        materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPM;
          materialGridType2 = GridType::OMEGAPM;

          materialGrid3 = GammaM;
          materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      gridType = GridType::HY;

      materialGrid = Mu;
      materialGridType = GridType::MU;
      materialModifier = PhysicsConst::Mu0;

      oppositeGrid1 = Ez;
      oppositeGrid2 = Ex;

      rightSideFunc = My;
      borderFunc = HyBorder;
      exactFunc = HyExact;

      if (usePML)
      {
        grid = By;
        gridType = GridType::BY;

        gridPML1 = B1y;
        gridPMLType1 = GridType::BY;

        gridPML2 = Hy;
        gridPMLType2 = GridType::HY;

        materialGrid = SigmaZ;
        materialGridType = GridType::SIGMAZ;

        materialGrid1 = Mu;
        materialGridType1 = GridType::MU;

        materialGrid4 = SigmaY;
        materialGridType4 = GridType::SIGMAY;

        materialGrid5 = SigmaX;
        materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPM;
          materialGridType2 = GridType::OMEGAPM;

          materialGrid3 = GammaM;
          materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      gridType = GridType::HZ;
      materialGrid = Mu;
      materialGridType = GridType::MU;
      materialModifier = PhysicsConst::Mu0;

      oppositeGrid1 = Ex;
      oppositeGrid2 = Ey;

      rightSideFunc = Mz;
      borderFunc = HzBorder;
      exactFunc = HzExact;

      if (usePML)
      {
        grid = Bz;
        gridType = GridType::BZ;

        gridPML1 = B1z;
        gridPMLType1 = GridType::BZ;

        gridPML2 = Hz;
        gridPMLType2 = GridType::HZ;

        materialGrid = SigmaX;
        materialGridType = GridType::SIGMAX;

        materialGrid1 = Mu;
        materialGridType1 = GridType::MU;

        materialGrid4 = SigmaZ;
        materialGridType4 = GridType::SIGMAZ;

        materialGrid5 = SigmaY;
        materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          materialGrid2 = OmegaPM;
          materialGridType2 = GridType::OMEGAPM;

          materialGrid3 = GammaM;
          materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (t > 0)
  {
    for (int i = start.getX (); i < end.getX (); ++i)
    {
      for (int j = start.getY (); j < end.getY (); ++j)
      {
        for (int k = start.getZ (); k < end.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          // TODO: add getTotalPositionDiff here, which will be called before loop
          GridCoordinate3D posAbs = grid->getTotalPosition (pos);
          // TODO: [possible] move 1D gridValues to 3D gridValues array
          FieldPointValue *valField = grid->getFieldPointValue (pos);

          FPValue material = yeeLayout->getMaterial (posAbs, gridType, materialGrid, materialGridType);

          GridCoordinate3D pos11;
          GridCoordinate3D pos12;
          GridCoordinate3D pos21;
          GridCoordinate3D pos22;

          GridCoordinateFP3D coordFP;
          FPValue timestep;

          FPValue k_mod;
          FPValue Ca;
          FPValue Cb;

          // TODO: add circuitElementDiff here, which will be called before loop
          // TODO: add coordFPDiff here, which will be called before loop
          switch (grid_type)
          {
            case (static_cast<uint8_t> (GridType::EX)):
            {
              pos11 = yeeLayout->getExCircuitElement (pos, LayoutDirection::DOWN);
              pos12 = yeeLayout->getExCircuitElement (pos, LayoutDirection::UP);
              pos21 = yeeLayout->getExCircuitElement (pos, LayoutDirection::BACK);
              pos22 = yeeLayout->getExCircuitElement (pos, LayoutDirection::FRONT);

              // TODO: do not invoke in case no right side
              coordFP = yeeLayout->getExCoordFP (posAbs);
              timestep = t;

              FPValue k_y = 1;
              k_mod = k_y;
              break;
            }
            case (static_cast<uint8_t> (GridType::EY)):
            {
              pos11 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::BACK);
              pos12 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::FRONT);
              pos21 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::LEFT);
              pos22 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::RIGHT);

              coordFP = yeeLayout->getEyCoordFP (posAbs);
              timestep = t;

              FPValue k_z = 1;
              k_mod = k_z;
              break;
            }
            case (static_cast<uint8_t> (GridType::EZ)):
            {
              pos11 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::LEFT);
              pos12 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::RIGHT);
              pos21 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::DOWN);
              pos22 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::UP);

              coordFP = yeeLayout->getEzCoordFP (posAbs);
              timestep = t;

              FPValue k_x = 1;
              k_mod = k_x;
              break;
            }
            case (static_cast<uint8_t> (GridType::HX)):
            {
              pos11 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::BACK);
              pos12 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::FRONT);
              pos21 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::DOWN);
              pos22 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::UP);

              coordFP = yeeLayout->getHxCoordFP (posAbs);
              timestep = t + 0.5;

              FPValue k_y = 1;
              k_mod = k_y;
              break;
            }
            case (static_cast<uint8_t> (GridType::HY)):
            {
              pos11 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::LEFT);
              pos12 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::RIGHT);
              pos21 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::BACK);
              pos22 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::FRONT);

              coordFP = yeeLayout->getHyCoordFP (posAbs);
              timestep = t + 0.5;

              FPValue k_z = 1;
              k_mod = k_z;
              break;
            }
            case (static_cast<uint8_t> (GridType::HZ)):
            {
              pos11 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::DOWN);
              pos12 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::UP);
              pos21 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::LEFT);
              pos22 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::RIGHT);

              coordFP = yeeLayout->getHzCoordFP (posAbs);
              timestep = t + 0.5;

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

          FieldPointValue *val11 = oppositeGrid1->getFieldPointValue (pos11);
          FieldPointValue *val12 = oppositeGrid1->getFieldPointValue (pos12);

          FieldPointValue *val21 = oppositeGrid2->getFieldPointValue (pos21);
          FieldPointValue *val22 = oppositeGrid2->getFieldPointValue (pos22);

          // TODO: separate previous grid and current
          FieldValue prev11 = val11->getPrevValue ();
          FieldValue prev12 = val12->getPrevValue ();

          FieldValue prev21 = val21->getPrevValue ();
          FieldValue prev22 = val22->getPrevValue ();

          if (solverSettings.getDoUseTFSF ())
          {
            // TODO: unify
            // calculateTFSF<grid_type> (posAbs, prev12, prev11, prev22, prev21, pos12, pos11, pos22, pos21);
            switch (grid_type)
            {
              case (static_cast<uint8_t> (GridType::EX)):
              {
                calculateExTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case (static_cast<uint8_t> (GridType::EY)):
              {
                calculateEyTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case (static_cast<uint8_t> (GridType::EZ)):
              {
                calculateEzTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case (static_cast<uint8_t> (GridType::HX)):
              {
                calculateHxTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case (static_cast<uint8_t> (GridType::HY)):
              {
                calculateHyTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case (static_cast<uint8_t> (GridType::HZ)):
              {
                calculateHzTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              default:
              {
                UNREACHABLE;
              }
            }
          }

          FieldValue prevRightSide = 0;
          if (rightSideFunc != NULLPTR)
          {
            prevRightSide = rightSideFunc (coordFP * gridStep, timestep * gridTimeStep);
          }

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

          valField->setCurValue (val);
        }
      }
    }

    if (usePML)
    {
      if (useMetamaterials)
      {
#ifdef TWO_TIME_STEPS
        for (int i = start.getX (); i < end.getX (); ++i)
        {
          for (int j = start.getY (); j < end.getY (); ++j)
          {
            for (int k = start.getZ (); k < end.getZ (); ++k)
            {
              GridCoordinate3D pos (i, j, k);
              GridCoordinate3D posAbs = grid->getTotalPosition (pos);
              FieldPointValue *valField = grid->getFieldPointValue (pos);
              FieldPointValue *valField1 = gridPML1->getFieldPointValue (pos);

              FPValue material1;
              FPValue material2;
              FPValue material = yeeLayout->getMetaMaterial (posAbs, gridType,
                                                             materialGrid1, materialGridType1,
                                                             materialGrid2, materialGridType2,
                                                             materialGrid3, materialGridType3,
                                                             material1, material2);

              /*
               * FIXME: precalculate coefficients
               */
              FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
              FPValue a1 = (4 + 2*gridTimeStep*material2) / A;
              FPValue a2 = -8 / A;
              FPValue a3 = (4 - 2*gridTimeStep*material2) / A;
              FPValue a4 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
              FPValue a5 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;

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
            }
          }
        }
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
      }

      for (int i = start.getX (); i < end.getX (); ++i)
      {
        for (int j = start.getY (); j < end.getY (); ++j)
        {
          for (int k = start.getZ (); k < end.getZ (); ++k)
          {
            GridCoordinate3D pos (i, j, k);
            GridCoordinate3D posAbs = gridPML2->getTotalPosition (pos);

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

            FPValue material1 = yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1);
            FPValue material4 = yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4);
            FPValue material5 = yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5);

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

            FieldValue val = calcFieldFromDOrB (valField->getPrevValue (),
                                                valField1->getCurValue (),
                                                valField1->getPrevValue (),
                                                Ca,
                                                Cb,
                                                Cc);

            valField->setCurValue (val);
          }
        }
      }
    }
  }

  if (borderFunc != NULLPTR)
  {
    for (int i = 0; i < grid->getSize ().getX (); ++i)
    {
      for (int j = 0; j < grid->getSize ().getY (); ++j)
      {
        for (int k = 0; k < grid->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = grid->getTotalPosition (pos);

          if (posAbs.getX () != 0 && posAbs.getX () != grid->getTotalSize ().getX () - 1
              && posAbs.getY () != 0 && posAbs.getY () != grid->getTotalSize ().getY () - 1
              && posAbs.getZ () != 0 && posAbs.getZ () != grid->getTotalSize ().getZ () - 1)
          {
            continue;
          }

          GridCoordinateFP3D realCoord;
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

          grid->getFieldPointValue (pos)->setCurValue (borderFunc (realCoord * gridStep, timestep * gridTimeStep));
        }
      }
    }
  }

  if (exactFunc != NULLPTR)
  {
#ifdef COMPLEX_FIELD_VALUES
    FPValue normRe = 0.0;
    FPValue normIm = 0.0;
    FPValue normMod = 0.0;

    FPValue maxRe = 0.0;
    FPValue maxIm = 0.0;
    FPValue maxMod = 0.0;
#else
    FPValue norm = 0.0;
    FPValue max = 0.0;
#endif

    for (int i = 0; i < grid->getSize ().getX (); ++i)
    {
      for (int j = 0; j < grid->getSize ().getY (); ++j)
      {
        for (int k = 0; k < grid->getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posAbs = grid->getTotalPosition (pos);

          GridCoordinateFP3D realCoord;
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
          FieldValue exact = exactFunc (realCoord * gridStep, timestep * gridTimeStep);

#ifdef COMPLEX_FIELD_VALUES
          FPValue modExact = sqrt (SQR (exact.real ()) + SQR (exact.imag ()));
          FPValue modNumerical = sqrt (SQR (numerical.real ()) + SQR (numerical.imag ()));

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
          norm += SQR (exact - numerical);

          FPValue exactAbs = fabs (exact);
          if (max < exactAbs)
          {
            max = exactAbs;
          }
#endif
        }
      }
    }

#ifdef COMPLEX_FIELD_VALUES
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
    normIm = sqrt (normIm / grid->getSize ().calculateTotalCoord ());
    normMod = sqrt (normMod / grid->getSize ().calculateTotalCoord ());
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( %.20f , %.20f ) = ( %.20f %% , %.20f %% ), module = %.20f = ( %.20f %% )\n",
      grid->getName ().c_str (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
    norm = sqrt (norm / grid->getSize ().calculateTotalCoord ());
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( %.20f ) = ( %.20f %% )\n",
      grid->getName ().c_str (), t, norm, norm * 100.0 / max);
#endif
  }
}

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */

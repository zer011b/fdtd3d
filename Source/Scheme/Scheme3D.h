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

  template <uint8_t grid_coord>
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

  FieldValue calcFieldPrecalc (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                               FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                               FPValue Ca, FPValue Cb, FPValue delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return Ca * prev + Cb * tmp;
  }

  FieldValue calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                        FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                        FPValue dt, FPValue delta, FPValue material)
  {
    return calcFieldPrecalc (prev, oppositeField12, oppositeField11, oppositeField22, oppositeField21, prevRightSide,
                             FPValue (1.0), dt / (material * delta), delta);
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

template<uint8_t grid_type>
void
Scheme3D::calculateFieldStep (time_step t, GridCoordinate3D start, GridCoordinate3D end)
{
  Grid<GridCoordinate3D> *grid = NULLPTR;
  GridType gridType;

  Grid<GridCoordinate3D> *materialGrid = NULLPTR;
  GridType materialGridType;

  Grid<GridCoordinate3D> *oppositeGrid1 = NULLPTR;
  Grid<GridCoordinate3D> *oppositeGrid2 = NULLPTR;

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
          GridCoordinate3D posAbs = grid->getTotalPosition (pos);

          FieldPointValue* valField = grid->getFieldPointValue (pos);

          FPValue material = yeeLayout->getMaterial (posAbs, gridType, materialGrid, materialGridType);

          GridCoordinate3D pos11;
          GridCoordinate3D pos12;
          GridCoordinate3D pos21;
          GridCoordinate3D pos22;

          GridCoordinateFP3D coordFP;
          FPValue timestep;

          switch (gridType)
          {
            case GridType::EX:
            {
              pos11 = yeeLayout->getExCircuitElement (pos, LayoutDirection::DOWN);
              pos12 = yeeLayout->getExCircuitElement (pos, LayoutDirection::UP);
              pos21 = yeeLayout->getExCircuitElement (pos, LayoutDirection::BACK);
              pos22 = yeeLayout->getExCircuitElement (pos, LayoutDirection::FRONT);

              coordFP = yeeLayout->getExCoordFP (posAbs);
              timestep = t;
              break;
            }
            case GridType::EY:
            {
              pos11 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::BACK);
              pos12 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::FRONT);
              pos21 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::LEFT);
              pos22 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::RIGHT);

              coordFP = yeeLayout->getEyCoordFP (posAbs);
              timestep = t;
              break;
            }
            case GridType::EZ:
            {
              pos11 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::LEFT);
              pos12 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::RIGHT);
              pos21 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::DOWN);
              pos22 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::UP);

              coordFP = yeeLayout->getEzCoordFP (posAbs);
              timestep = t;
              break;
            }
            case GridType::HX:
            {
              pos11 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::BACK);
              pos12 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::FRONT);
              pos21 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::DOWN);
              pos22 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::UP);

              coordFP = yeeLayout->getHxCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::HY:
            {
              pos11 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::LEFT);
              pos12 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::RIGHT);
              pos21 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::BACK);
              pos22 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::FRONT);

              coordFP = yeeLayout->getHyCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::HZ:
            {
              pos11 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::DOWN);
              pos12 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::UP);
              pos21 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::LEFT);
              pos22 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::RIGHT);

              coordFP = yeeLayout->getHzCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            default:
            {
              UNREACHABLE;
            }
          }

          FieldPointValue* val11 = oppositeGrid1->getFieldPointValue (pos11);
          FieldPointValue* val12 = oppositeGrid1->getFieldPointValue (pos12);

          FieldPointValue* val21 = oppositeGrid2->getFieldPointValue (pos21);
          FieldPointValue* val22 = oppositeGrid2->getFieldPointValue (pos22);

          FieldValue prev11 = val11->getPrevValue ();
          FieldValue prev12 = val12->getPrevValue ();

          FieldValue prev21 = val21->getPrevValue ();
          FieldValue prev22 = val22->getPrevValue ();

          if (solverSettings.getDoUseTFSF ())
          {
            // TODO: unify
            // calculateTFSF<grid_type> (posAbs, prev12, prev11, prev22, prev21, pos12, pos11, pos22, pos21);
            switch (gridType)
            {
              case GridType::EX:
              {
                calculateExTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case GridType::EY:
              {
                calculateEyTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case GridType::EZ:
              {
                calculateEzTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case GridType::HX:
              {
                calculateHxTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case GridType::HY:
              {
                calculateHyTFSF (posAbs, prev12, prev11, prev22, prev21, pos11, pos12, pos21, pos22);
                break;
              }
              case GridType::HZ:
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

          FieldValue val = calcField (valField->getPrevValue (),
                                      prev12,
                                      prev11,
                                      prev22,
                                      prev21,
                                      prevRightSide,
                                      gridTimeStep,
                                      gridStep,
                                      material * materialModifier);

          valField->setCurValue (val);
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
          switch (gridType)
          {
            case GridType::EX:
            {
              realCoord = yeeLayout->getExCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::EY:
            {
              realCoord = yeeLayout->getEyCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::EZ:
            {
              realCoord = yeeLayout->getEzCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::HX:
            {
              realCoord = yeeLayout->getHxCoordFP (posAbs);
              timestep = t + 1.0;
              break;
            }
            case GridType::HY:
            {
              realCoord = yeeLayout->getHyCoordFP (posAbs);
              timestep = t + 1.0;
              break;
            }
            case GridType::HZ:
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
          switch (gridType)
          {
            case GridType::EX:
            {
              realCoord = yeeLayout->getExCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::EY:
            {
              realCoord = yeeLayout->getEyCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::EZ:
            {
              realCoord = yeeLayout->getEzCoordFP (posAbs);
              timestep = t + 0.5;
              break;
            }
            case GridType::HX:
            {
              realCoord = yeeLayout->getHxCoordFP (posAbs);
              timestep = t + 1.0;
              break;
            }
            case GridType::HY:
            {
              realCoord = yeeLayout->getHyCoordFP (posAbs);
              timestep = t + 1.0;
              break;
            }
            case GridType::HZ:
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

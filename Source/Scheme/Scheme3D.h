#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "Scheme.h"
#include "ParallelYeeGridLayout.h"

#ifdef GRID_3D

#if defined (PARALLEL_GRID)
typedef ParallelGrid FieldGrid;
#else
typedef Grid<GridCoordinate3D> FieldGrid;
#endif

class Scheme3D: public Scheme
{
  YeeGridLayout *yeeLayout;

  FieldGrid Ex;
  FieldGrid Ey;
  FieldGrid Ez;
  FieldGrid Hx;
  FieldGrid Hy;
  FieldGrid Hz;

  FieldGrid Dx;
  FieldGrid Dy;
  FieldGrid Dz;
  FieldGrid Bx;
  FieldGrid By;
  FieldGrid Bz;

  FieldGrid D1x;
  FieldGrid D1y;
  FieldGrid D1z;
  FieldGrid B1x;
  FieldGrid B1y;
  FieldGrid B1z;

  FieldGrid ExAmplitude;
  FieldGrid EyAmplitude;
  FieldGrid EzAmplitude;
  FieldGrid HxAmplitude;
  FieldGrid HyAmplitude;
  FieldGrid HzAmplitude;

  FieldGrid Eps;
  FieldGrid Mu;

  FieldGrid SigmaX;
  FieldGrid SigmaY;
  FieldGrid SigmaZ;

  FieldGrid OmegaPE;
  FieldGrid GammaE;

  FieldGrid OmegaPM;
  FieldGrid GammaM;

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

  bool calculateAmplitude;

  time_step amplitudeStepLimit;

  bool usePML;

  bool useTFSF;

  Grid<GridCoordinate1D> EInc;
  Grid<GridCoordinate1D> HInc;

  bool useMetamaterials;

  bool dumpRes;

  bool useNTFF;

  GridCoordinate3D leftNTFF;
  GridCoordinate3D rightNTFF;

private:

  void calculateExStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEzStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateExStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  FieldValue approximateIncidentWave (GridCoordinateFP3D, FPValue, Grid<GridCoordinate1D> &);
  FieldValue approximateIncidentWaveE (GridCoordinateFP3D);
  FieldValue approximateIncidentWaveH (GridCoordinateFP3D);

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

  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

  //void makeGridScattered (Grid<GridCoordinate3D> &);

public:

  virtual void performSteps () CXX11_OVERRIDE;

  void initScheme (FPValue, FPValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  Scheme3D (ParallelYeeGridLayout *layout,
            const GridCoordinate3D& totSize,
            const GridCoordinate3D& bufSize,
            time_step tStep,
            bool calcAmp = false,
            time_step ampStep = 0,
            bool doUsePML = false,
            bool doUseTFSF = false,
            bool doUseMetamaterials = false,
            bool doUseNTFF = false,
            bool doDumpRes = false) :
    yeeLayout (layout),
    Ex (layout->getExSize (), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    Ey (layout->getEySize (), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    Ez (layout->getEzSize (), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    Hx (layout->getHxSize (), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    Hy (layout->getHySize (), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    Hz (layout->getHzSize (), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    Dx (layout->getExSize (), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    Dy (layout->getEySize (), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    Dz (layout->getEzSize (), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    Bx (layout->getHxSize (), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    By (layout->getHySize (), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    Bz (layout->getHzSize (), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    D1x (layout->getExSize (), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    D1y (layout->getEySize (), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    D1z (layout->getEzSize (), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    B1x (layout->getHxSize (), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    B1y (layout->getHySize (), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    B1z (layout->getHzSize (), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    ExAmplitude (layout->getExSize (), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    EyAmplitude (layout->getEySize (), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    EzAmplitude (layout->getEzSize (), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    HxAmplitude (layout->getHxSize (), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    HyAmplitude (layout->getHySize (), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    HzAmplitude (layout->getHzSize (), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    Eps (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    Mu (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getMuSizeForCurNode (), layout->getMuCoreSizePerNode ()),
    OmegaPE (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    GammaE (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    OmegaPM (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    GammaM (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaX (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaY (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaZ (layout->getEpsSize (), bufSize + GridCoordinate3D (1, 1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    sourceWaveLength (0),
    sourceFrequency (0),
    courantNum (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0),
    useMetamaterials (doUseMetamaterials),
    dumpRes (doDumpRes),
    useNTFF (doUseNTFF),
    leftNTFF (GridCoordinate3D (13, 13, 13)),
    rightNTFF (layout->getEzSize () - leftNTFF + GridCoordinate3D (1,1,1))
#else
  Scheme3D (YeeGridLayout *layout,
            const GridCoordinate3D& totSize,
            time_step tStep,
            bool calcAmp = false,
            time_step ampStep = 0,
            bool doUsePML = false,
            bool doUseTFSF = false,
            bool doUseMetamaterials = false,
            bool doUseNTFF = false,
            bool doDumpRes = false) :
    yeeLayout (layout),
    Ex (layout->getExSize (), 0),
    Ey (layout->getEySize (), 0),
    Ez (layout->getEzSize (), 0),
    Hx (layout->getHxSize (), 0),
    Hy (layout->getHySize (), 0),
    Hz (layout->getHzSize (), 0),
    Dx (layout->getExSize (), 0),
    Dy (layout->getEySize (), 0),
    Dz (layout->getEzSize (), 0),
    Bx (layout->getHxSize (), 0),
    By (layout->getHySize (), 0),
    Bz (layout->getHzSize (), 0),
    D1x (layout->getExSize (), 0),
    D1y (layout->getEySize (), 0),
    D1z (layout->getEzSize (), 0),
    B1x (layout->getHxSize (), 0),
    B1y (layout->getHySize (), 0),
    B1z (layout->getHzSize (), 0),
    ExAmplitude (layout->getExSize (), 0),
    EyAmplitude (layout->getEySize (), 0),
    EzAmplitude (layout->getEzSize (), 0),
    HxAmplitude (layout->getHxSize (), 0),
    HyAmplitude (layout->getHySize (), 0),
    HzAmplitude (layout->getHzSize (), 0),
    Eps (layout->getEpsSize (), 0),
    Mu (layout->getEpsSize (), 0),
    OmegaPE (layout->getEpsSize (), 0),
    GammaE (layout->getEpsSize (), 0),
    OmegaPM (layout->getEpsSize (), 0),
    GammaM (layout->getEpsSize (), 0),
    SigmaX (layout->getEpsSize (), 0),
    SigmaY (layout->getEpsSize (), 0),
    SigmaZ (layout->getEpsSize (), 0),
    sourceWaveLength (0),
    sourceFrequency (0),
    courantNum (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY () + totSize.getZ ())), 0),
    useMetamaterials (doUseMetamaterials),
    dumpRes (doDumpRes),
    useNTFF (doUseNTFF),
    leftNTFF (GridCoordinate3D (13, 13, 13)),
    rightNTFF (layout->getEzSize () - leftNTFF + GridCoordinate3D (1,1,1))
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF && yeeLayout->getSizeTFSF () != GridCoordinate3D (0, 0, 0)));

    ASSERT (!doUsePML || (doUsePML && (yeeLayout->getSizePML () != GridCoordinate3D (0, 0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);

#ifdef COMPLEX_FIELD_VALUES
    ASSERT (!calculateAmplitude);
#endif /* COMPLEX_FIELD_VALUES */
  }

  ~Scheme3D ()
  {
  }

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
  NPair ntffN_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);
  NPair ntffN_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);
  NPair ntffN_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);

  NPair ntffL_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);
  NPair ntffL_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);
  NPair ntffL_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);

  NPair ntffN (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &,
               Grid<GridCoordinate3D> &);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &,
               Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &, Grid<GridCoordinate3D> &);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);

  FPValue getMaterial (FieldGrid, GridCoordinate3D, GridType, GridType);
  FPValue getMaterial (FieldGrid, GridType);
};

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */

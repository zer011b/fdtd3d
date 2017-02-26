#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "Scheme.h"
#include "ParallelYeeGridLayout.h"

#ifdef GRID_3D

class Scheme3D: public Scheme
{
  YeeGridLayout *yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ex;
  ParallelGrid Ey;
  ParallelGrid Ez;
  ParallelGrid Hx;
  ParallelGrid Hy;
  ParallelGrid Hz;

  ParallelGrid Dx;
  ParallelGrid Dy;
  ParallelGrid Dz;
  ParallelGrid Bx;
  ParallelGrid By;
  ParallelGrid Bz;

  ParallelGrid D1x;
  ParallelGrid D1y;
  ParallelGrid D1z;
  ParallelGrid B1x;
  ParallelGrid B1y;
  ParallelGrid B1z;

  ParallelGrid ExAmplitude;
  ParallelGrid EyAmplitude;
  ParallelGrid EzAmplitude;
  ParallelGrid HxAmplitude;
  ParallelGrid HyAmplitude;
  ParallelGrid HzAmplitude;

  ParallelGrid Eps;
  ParallelGrid Mu;

  ParallelGrid SigmaX;
  ParallelGrid SigmaY;
  ParallelGrid SigmaZ;

  ParallelGrid OmegaPE;
  ParallelGrid GammaE;

  ParallelGrid OmegaPM;
  ParallelGrid GammaM;
#else
  Grid<GridCoordinate3D> Ex;
  Grid<GridCoordinate3D> Ey;
  Grid<GridCoordinate3D> Ez;
  Grid<GridCoordinate3D> Hx;
  Grid<GridCoordinate3D> Hy;
  Grid<GridCoordinate3D> Hz;

  Grid<GridCoordinate3D> Dx;
  Grid<GridCoordinate3D> Dy;
  Grid<GridCoordinate3D> Dz;
  Grid<GridCoordinate3D> Bx;
  Grid<GridCoordinate3D> By;
  Grid<GridCoordinate3D> Bz;

  Grid<GridCoordinate3D> D1x;
  Grid<GridCoordinate3D> D1y;
  Grid<GridCoordinate3D> D1z;
  Grid<GridCoordinate3D> B1x;
  Grid<GridCoordinate3D> B1y;
  Grid<GridCoordinate3D> B1z;

  Grid<GridCoordinate3D> ExAmplitude;
  Grid<GridCoordinate3D> EyAmplitude;
  Grid<GridCoordinate3D> EzAmplitude;
  Grid<GridCoordinate3D> HxAmplitude;
  Grid<GridCoordinate3D> HyAmplitude;
  Grid<GridCoordinate3D> HzAmplitude;

  Grid<GridCoordinate3D> Eps;
  Grid<GridCoordinate3D> Mu;

  Grid<GridCoordinate3D> SigmaX;
  Grid<GridCoordinate3D> SigmaY;
  Grid<GridCoordinate3D> SigmaZ;

  Grid<GridCoordinate3D> OmegaPE;
  Grid<GridCoordinate3D> GammaE;

  Grid<GridCoordinate3D> OmegaPM;
  Grid<GridCoordinate3D> GammaM;
#endif

  // Wave parameters
  FPValue sourceWaveLength;
  FPValue sourceFrequency;

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

  FPValue incidentWaveAngle1; // Teta
  FPValue incidentWaveAngle2; // Phi
  FPValue incidentWaveAngle3; // Psi

  bool useMetamaterials;

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

  void performExSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHzSteps (time_step, GridCoordinate3D, GridCoordinate3D);

  void performNSteps (time_step, time_step, int);
  void performAmplitudeSteps (time_step, int);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

public:

  virtual void performSteps (int) CXX11_OVERRIDE;

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
            FPValue angleIncWave1 = 0.0,
            FPValue angleIncWave2 = 0.0,
            FPValue angleIncWave3 = 0.0,
            bool doUseMetamaterials = false) :
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
    incidentWaveAngle1 (angleIncWave1),
    incidentWaveAngle2 (angleIncWave2),
    incidentWaveAngle3 (angleIncWave3),
    useMetamaterials (doUseMetamaterials)
#else
  Scheme3D (YeeGridLayout *layout,
            const GridCoordinate3D& totSize,
            time_step tStep,
            bool calcAmp = false,
            time_step ampStep = 0,
            bool doUsePML = false,
            bool doUseTFSF = false,
            FPValue angleIncWave1 = 0.0,
            FPValue angleIncWave2 = 0.0,
            FPValue angleIncWave3 = 0.0,
            bool doUseMetamaterials = false) :
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
    incidentWaveAngle1 (angleIncWave1),
    incidentWaveAngle2 (angleIncWave2),
    incidentWaveAngle3 (angleIncWave3),
    useMetamaterials (doUseMetamaterials)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && incidentWaveAngle1 == PhysicsConst::Pi / 2
                && (incidentWaveAngle2 == PhysicsConst::Pi / 4 || incidentWaveAngle2 == 0)
                && yeeLayout->getSizeTFSF ()) != GridCoordinate3D (0, 0, 0)));

    ASSERT (!doUsePML || (doUsePML && (yeeLayout->getSizePML ()) != GridCoordinate3D (0, 0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);

#ifdef COMPLEX_FIELD_VALUES
    ASSERT (!calculateAmplitude);
#endif /* COMPLEX_FIELD_VALUES */
  }

  ~Scheme3D ()
  {
  }
};

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */

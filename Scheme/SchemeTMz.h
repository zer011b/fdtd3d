#ifndef SCHEME_TMZ_H
#define SCHEME_TMZ_H

#include "Scheme.h"
#include "ParallelGrid.h"
#include "Grid.h"
#include "YeeGridLayout.h"
#include "PhysicsConst.h"

#ifdef GRID_2D

class SchemeTMz: public Scheme
{
  YeeGridLayout yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ez;
  ParallelGrid Hx;
  ParallelGrid Hy;

  ParallelGrid Dz;
  ParallelGrid Bx;
  ParallelGrid By;

  ParallelGrid D1z;
  ParallelGrid B1x;
  ParallelGrid B1y;

  ParallelGrid EzAmplitude;
  ParallelGrid HxAmplitude;
  ParallelGrid HyAmplitude;

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
  Grid<GridCoordinate2D> Ez;
  Grid<GridCoordinate2D> Hx;
  Grid<GridCoordinate2D> Hy;

  Grid<GridCoordinate2D> Dz;
  Grid<GridCoordinate2D> Bx;
  Grid<GridCoordinate2D> By;

  Grid<GridCoordinate2D> D1z;
  Grid<GridCoordinate2D> B1x;
  Grid<GridCoordinate2D> B1y;

  Grid<GridCoordinate2D> EzAmplitude;
  Grid<GridCoordinate2D> HxAmplitude;
  Grid<GridCoordinate2D> HyAmplitude;

  Grid<GridCoordinate2D> Eps;
  Grid<GridCoordinate2D> Mu;

  Grid<GridCoordinate2D> SigmaX;
  Grid<GridCoordinate2D> SigmaY;
  Grid<GridCoordinate2D> SigmaZ;

  Grid<GridCoordinate2D> OmegaPE;
  Grid<GridCoordinate2D> GammaE;

  Grid<GridCoordinate2D> OmegaPM;
  Grid<GridCoordinate2D> GammaM;
#endif

  // Wave parameters
  FPValue waveLength;
  FPValue stepWaveLength;
  FPValue frequency;

  // dx
  FPValue gridStep;

  // dt
  FPValue gridTimeStep;

  time_step totalStep;

  int process;

  bool calculateAmplitude;

  time_step amplitudeStepLimit;

  bool usePML;

  bool useTFSF;

  Grid<GridCoordinate1D> EInc;
  Grid<GridCoordinate1D> HInc;

  FPValue incidentWaveAngle;

  bool useMetamaterials;

private:

  void calculateEzStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performNSteps (time_step, time_step, int);
  void performAmplitudeSteps (time_step, int);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

public:

#ifdef CXX11_ENABLED
  virtual void performSteps (int) override;
#else
  virtual void performSteps (int);
#endif

  void initScheme (FPValue, FPValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  void initProcess (int);
#endif

#if defined (PARALLEL_GRID)
  SchemeTMz (const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSizeL,
             const GridCoordinate2D& bufSizeR,
             const int curProcess,
             const int totalProc,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             GridCoordinate2D sizePML = GridCoordinate2D (0, 0),
             bool doUseTFSF = false,
             GridCoordinate2D sizeScatteredZone = GridCoordinate2D (0, 0),
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, PhysicsConst::Pi / 2, angleIncWave, 0),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dz (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Bx (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    By (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    D1z (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    B1x (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    B1y (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    EzAmplitude (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HxAmplitude (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HyAmplitude (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Eps (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    Mu (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    OmegaPE (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    GammaE (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    OmegaPM (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    GammaM (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaX (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaY (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaZ (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (curProcess),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    useMetamaterials (doUseMetamaterials)
#else
  SchemeTMz (const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             GridCoordinate2D sizePML = GridCoordinate2D (0, 0),
             bool doUseTFSF = false,
             GridCoordinate2D sizeScatteredZone = GridCoordinate2D (0, 0),
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, PhysicsConst::Pi / 2, angleIncWave, 0),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), 0),
    Dz (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Bx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    By (shrinkCoord (yeeLayout.getHySize ()), 0),
    D1z (shrinkCoord (yeeLayout.getEzSize ()), 0),
    B1x (shrinkCoord (yeeLayout.getHxSize ()), 0),
    B1y (shrinkCoord (yeeLayout.getHySize ()), 0),
    EzAmplitude (shrinkCoord (yeeLayout.getEzSize ()), 0),
    HxAmplitude (shrinkCoord (yeeLayout.getHxSize ()), 0),
    HyAmplitude (shrinkCoord (yeeLayout.getHySize ()), 0),
    Eps (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    Mu (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    OmegaPE (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    GammaE (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    OmegaPM (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    GammaM (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    SigmaX (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    SigmaY (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    SigmaZ (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (0),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    useMetamaterials (doUseMetamaterials)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && (incidentWaveAngle == PhysicsConst::Pi / 4 || incidentWaveAngle == 0)
                && sizeScatteredZone != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (sizePML != GridCoordinate2D (0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);
  }

  ~SchemeTMz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TMZ_H */

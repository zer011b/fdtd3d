#ifndef SCHEME_TMZ_H
#define SCHEME_TMZ_H

#include "Scheme.h"
#include "GridInterface.h"
#include "ParallelYeeGridLayout.h"
#include "PhysicsConst.h"

#ifdef GRID_2D

class SchemeTMz: public Scheme
{
  YeeGridLayout *yeeLayout;

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

  virtual void performSteps (int) CXX11_OVERRIDE;

  void initScheme (FPValue, FPValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  SchemeTMz (ParallelYeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false) :
    yeeLayout (layout),
    Ez (shrinkCoord (layout->getEzSize ()), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    Hx (shrinkCoord (layout->getHxSize ()), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    Hy (shrinkCoord (layout->getHySize ()), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    Dz (shrinkCoord (layout->getEzSize ()), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    Bx (shrinkCoord (layout->getHxSize ()), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    By (shrinkCoord (layout->getHySize ()), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    D1z (shrinkCoord (layout->getEzSize ()), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    B1x (shrinkCoord (layout->getHxSize ()), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    B1y (shrinkCoord (layout->getHySize ()), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    EzAmplitude (shrinkCoord (layout->getEzSize ()), bufSize, 0, layout->getEzSizeForCurNode (), layout->getEzCoreSizePerNode ()),
    HxAmplitude (shrinkCoord (layout->getHxSize ()), bufSize, 0, layout->getHxSizeForCurNode (), layout->getHxCoreSizePerNode ()),
    HyAmplitude (shrinkCoord (layout->getHySize ()), bufSize, 0, layout->getHySizeForCurNode (), layout->getHyCoreSizePerNode ()),
    Eps (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    Mu (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getMuSizeForCurNode (), layout->getMuCoreSizePerNode ()),
    OmegaPE (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    GammaE (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    OmegaPM (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    GammaM (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaX (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaY (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaZ (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
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
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    useMetamaterials (doUseMetamaterials)
#else
  SchemeTMz (YeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false) :
    yeeLayout (layout),
    Ez (shrinkCoord (layout->getEzSize ()), 0),
    Hx (shrinkCoord (layout->getHxSize ()), 0),
    Hy (shrinkCoord (layout->getHySize ()), 0),
    Dz (shrinkCoord (layout->getEzSize ()), 0),
    Bx (shrinkCoord (layout->getHxSize ()), 0),
    By (shrinkCoord (layout->getHySize ()), 0),
    D1z (shrinkCoord (layout->getEzSize ()), 0),
    B1x (shrinkCoord (layout->getHxSize ()), 0),
    B1y (shrinkCoord (layout->getHySize ()), 0),
    EzAmplitude (shrinkCoord (layout->getEzSize ()), 0),
    HxAmplitude (shrinkCoord (layout->getHxSize ()), 0),
    HyAmplitude (shrinkCoord (layout->getHySize ()), 0),
    Eps (shrinkCoord (layout->getEpsSize ()), 0),
    Mu (shrinkCoord (layout->getEpsSize ()), 0),
    OmegaPE (shrinkCoord (layout->getEpsSize ()), 0),
    GammaE (shrinkCoord (layout->getEpsSize ()), 0),
    OmegaPM (shrinkCoord (layout->getEpsSize ()), 0),
    GammaM (shrinkCoord (layout->getEpsSize ()), 0),
    SigmaX (shrinkCoord (layout->getEpsSize ()), 0),
    SigmaY (shrinkCoord (layout->getEpsSize ()), 0),
    SigmaZ (shrinkCoord (layout->getEpsSize ()), 0),
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
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    useMetamaterials (doUseMetamaterials)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && (incidentWaveAngle == PhysicsConst::Pi / 4 || incidentWaveAngle == 0)
                && shrinkCoord (yeeLayout->getSizeTFSF ()) != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (shrinkCoord (yeeLayout->getSizePML ()) != GridCoordinate2D (0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);

#ifdef COMPLEX_FIELD_VALUES
    ASSERT (!calculateAmplitude);
#endif /* COMPLEX_FIELD_VALUES */
  }

  ~SchemeTMz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TMZ_H */

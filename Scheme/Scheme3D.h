// #ifndef SCHEME_3D_H
// #define SCHEME_3D_H
//
// #include "Scheme.h"
// #include "ParallelGrid.h"
// #include "Grid.h"
//
// class Scheme3D: public Scheme
// {
// #if defined (PARALLEL_GRID)
//   ParallelGrid Ex;
//   ParallelGrid Ey;
//   ParallelGrid Ez;
//   ParallelGrid Hx;
//   ParallelGrid Hy;
//   ParallelGrid Hz;
//
//   ParallelGrid Eps;
//   ParallelGrid Mu;
// #else
//   Grid<GridCoordinate3D> Ex;
//   Grid<GridCoordinate3D> Ey;
//   Grid<GridCoordinate3D> Ez;
//   Grid<GridCoordinate3D> Hx;
//   Grid<GridCoordinate3D> Hy;
//   Grid<GridCoordinate3D> Hz;
//
//   Grid<GridCoordinate3D> Eps;
//   Grid<GridCoordinate3D> Mu;
// #endif
//
//   // Wave parameters
//   FieldValue waveLength;
//   FieldValue stepWaveLength;
//   FieldValue frequency;
//
//   // dx
//   FieldValue gridStep;
//
//   // dt
//   FieldValue gridTimeStep;
//
//   uint32_t totalStep;
//
//   int process;
//
// public:
//
// #ifdef CXX11_ENABLED
//   virtual void performSteps () override;
// #else
//   virtual void performSteps ();
// #endif
//
//   void initScheme (FieldValue, FieldValue);
//
//   void initGrids ();
//
// #if defined (PARALLEL_GRID)
//   void initProcess (int);
// #endif
//
// #if defined (PARALLEL_GRID)
//   Scheme3D (const GridCoordinate3D& totSize,
//             const GridCoordinate3D& bufSizeL, const GridCoordinate3D& bufSizeR,
//             const int process, const int totalProc, uint32_t tStep) :
//     Ex (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Ey (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Ez (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Hx (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Hy (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Hz (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Eps (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     Mu (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
//     waveLength (0),
//     stepWaveLength (0),
//     frequency (0),
//     gridStep (0),
//     gridTimeStep (0),
//     totalStep (tStep)
//   {
//   }
// #else
//   Scheme3D (const GridCoordinate3D& totSize, uint32_t tStep) :
//     Ex (totSize, 0),
//     Ey (totSize, 0),
//     Ez (totSize, 0),
//     Hx (totSize, 0),
//     Hy (totSize, 0),
//     Hz (totSize, 0),
//     Eps (totSize, 0),
//     Mu (totSize, 0),
//     waveLength (0),
//     stepWaveLength (0),
//     frequency (0),
//     gridStep (0),
//     gridTimeStep (0),
//     totalStep (tStep)
//   {
//   }
// #endif
//
//   ~Scheme3D ()
//   {
//   }
// };
//
// #endif /* SCHEME_3D_H */

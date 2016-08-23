#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "Kernels.h"
#include "PhysicsConst.h"
#include "Scheme3D.h"
#include "GridLayout.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_3D

extern YeeGridLayout yeeLayout;
extern PhysicsConst PhConst;

void
Scheme3D::performSteps ()
{
// #if defined (CUDA_ENABLED)
//
//   int size = Ez.getSize().calculateTotalCoord();
//
//   FieldValue *tmp_Ez = new FieldValue [size];
//   FieldValue *tmp_Hx = new FieldValue [size];
//   FieldValue *tmp_Hy = new FieldValue [size];
//
//   FieldValue *tmp_Ez_prev = new FieldValue [size];
//   FieldValue *tmp_Hx_prev = new FieldValue [size];
//   FieldValue *tmp_Hy_prev = new FieldValue [size];
//
//   FieldValue *tmp_eps = new FieldValue [size];
//   FieldValue *tmp_mu = new FieldValue [size];
//
//   time_step t = 0;
//
// #ifdef PARALLEL_GRID
//   ParallelGridCoordinate bufSize = Ez.getBufferSize ();
//   time_step tStep = bufSize.getX ();
// #else
//   time_step tStep = totalStep;
// #endif
//
//   while (t < totalStep)
//   {
//     for (int i = 0; i < size; ++i)
//     {
//       FieldPointValue* valEz = Ez.getFieldPointValue (i);
//       tmp_Ez[i] = valEz->getCurValue ();
//       tmp_Ez_prev[i] = valEz->getPrevValue ();
//
//       FieldPointValue* valHx = Hx.getFieldPointValue (i);
//       tmp_Hx[i] = valHx->getCurValue ();
//       tmp_Hx_prev[i] = valHx->getPrevValue ();
//
//       FieldPointValue* valHy = Hy.getFieldPointValue (i);
//       tmp_Hy[i] = valHy->getCurValue ();
//       tmp_Hy_prev[i] = valHy->getPrevValue ();
//
//       FieldPointValue *valEps = Eps.getFieldPointValue (i);
//       tmp_eps[i] = valEps->getCurValue ();
//
//       FieldPointValue *valMu = Mu.getFieldPointValue (i);
//       tmp_mu[i] = valMu->getCurValue ();
//     }
//
//     CudaExitStatus exitStatus;
//     cudaExecute2DTMzSteps (&exitStatus,
//                            tmp_Ez, tmp_Hx, tmp_Hy,
//                            tmp_Ez_prev, tmp_Hx_prev, tmp_Hy_prev,
//                            tmp_eps, tmp_mu,
//                            gridTimeStep, gridStep,
//                            Ez.getSize ().getX (), Ez.getSize ().getY (),
//                            0, tStep, Ez.getSize ().getX () / 16, Ez.getSize ().getY () / 16, 16, 16);
//
//     ASSERT (exitStatus == CUDA_OK);
//
//     for (int i = 0; i < size; ++i)
//     {
//       /*if (tmp_Ez[i] != 0 || tmp_Ez_prev[i] != 0 ||
//           tmp_Hx[i] != 0 || tmp_Hx_prev[i] != 0 ||
//           tmp_Hy[i] != 0 || tmp_Hy_prev[i] != 0)
//       {
//         printf ("%d !!!!! %f %f %f %f %f %f\n", i, tmp_Ez[i], tmp_Ez_prev[i], tmp_Hx[i], tmp_Hx_prev[i], tmp_Hy[i], tmp_Hy_prev[i]);
//       }*/
//       FieldPointValue* valEz = Ez.getFieldPointValue (i);
//       valEz->setCurValue (tmp_Ez[i]);
//       valEz->setPrevValue (tmp_Ez_prev[i]);
//
//       FieldPointValue* valHx = Hx.getFieldPointValue (i);
//       valHx->setCurValue (tmp_Hx[i]);
//       valHx->setPrevValue (tmp_Hx_prev[i]);
//
//       FieldPointValue* valHy = Hy.getFieldPointValue (i);
//       valHy->setCurValue (tmp_Hy[i]);
//       valHy->setPrevValue (tmp_Hy_prev[i]);
//     }
//
// #if defined (PARALLEL_GRID)
//     Ez.share ();
//     Hx.share ();
//     Hy.share ();
// #endif /* PARALLEL_GRID */
//
//     t += tStep;
//   }
//
//   delete[] tmp_Ez;
//   delete[] tmp_Hx;
//   delete[] tmp_Hy;
//
//   delete[] tmp_Ez_prev;
//   delete[] tmp_Hx_prev;
//   delete[] tmp_Hy_prev;
//
//   delete[] tmp_eps;
//   delete[] tmp_mu;
//
// #if defined (PARALLEL_GRID)
//   if (process == 0)
// #endif
//   {
//     BMPDumper<GridCoordinate2D> dumper;
//     dumper.init (totalStep, ALL);
//     dumper.dumpGrid (Ez);
//   }
//
// #else /* CUDA_ENABLED */

  for (int t = 0; t < totalStep; ++t)
  {
    for (int i = 0; i < Ex.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ex.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ex.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posDown = yeeLayout.getExCircuitElement (pos, LayoutDirection::DOWN);
          GridCoordinate3D posUp = yeeLayout.getExCircuitElement (pos, LayoutDirection::UP);
          GridCoordinate3D posBack = yeeLayout.getExCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posFront = yeeLayout.getExCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valEx = Ex.getFieldPointValue (pos);
          FieldPointValue* valEps = Eps.getFieldPointValue (pos);

          FieldPointValue* valHz1 = Hz.getFieldPointValue (posUp);
          FieldPointValue* valHz2 = Hz.getFieldPointValue (posDown);

          FieldPointValue* valHy1 = Hy.getFieldPointValue (posBack);
          FieldPointValue* valHy2 = Hy.getFieldPointValue (posFront);

          FieldValue val = calculateEx_3D (valEx->getPrevValue (),
                                           valHz1->getPrevValue (),
                                           valHz2->getPrevValue (),
                                           valHy1->getPrevValue (),
                                           valHy2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valEps->getCurValue ());

          FieldPointValue* tmp = Ex.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

    for (int i = 0; i < Ey.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ey.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ey.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getEyCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getEyCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posBack = yeeLayout.getEyCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posFront = yeeLayout.getEyCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valEy = Ey.getFieldPointValue (pos);
          FieldPointValue* valEps = Eps.getFieldPointValue (pos);

          FieldPointValue* valHx1 = Hx.getFieldPointValue (posFront);
          FieldPointValue* valHx2 = Hx.getFieldPointValue (posBack);

          FieldPointValue* valHz1 = Hz.getFieldPointValue (posLeft);
          FieldPointValue* valHz2 = Hz.getFieldPointValue (posRight);

          FieldValue val = calculateEy_3D (valEy->getPrevValue (),
                                           valHx1->getPrevValue (),
                                           valHx2->getPrevValue (),
                                           valHz1->getPrevValue (),
                                           valHz2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valEps->getCurValue ());

          FieldPointValue* tmp = Ey.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

    for (int i = 0; i < Ez.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ez.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Ez.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getEzCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getEzCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posDown = yeeLayout.getEzCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posUp = yeeLayout.getEzCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valEz = Ez.getFieldPointValue (pos);
          FieldPointValue* valEps = Eps.getFieldPointValue (pos);

          FieldPointValue* valHy1 = Hy.getFieldPointValue (posRight);
          FieldPointValue* valHy2 = Hy.getFieldPointValue (posLeft);

          FieldPointValue* valHx1 = Hx.getFieldPointValue (posDown);
          FieldPointValue* valHx2 = Hx.getFieldPointValue (posUp);

          FieldValue val = calculateEz_3D (valEz->getPrevValue (),
                                           valHy1->getPrevValue (),
                                           valHy2->getPrevValue (),
                                           valHx1->getPrevValue (),
                                           valHx2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valEps->getCurValue ());

          FieldPointValue* tmp = Ez.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

#if defined (PARALLEL_GRID)
    if (process == 0)
#endif
    {
      GridCoordinate3D pos (Ez.getSize ().getX () / 2, Ez.getSize ().getY () / 2, Ez.getSize ().getZ () / 2);
      FieldPointValue* tmp = Ez.getFieldPointValue (pos);
      tmp->setCurValue (cos (t * 3.1415 / 12));
    }

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();
    Ez.nextTimeStep ();

    for (int i = 0; i < Hx.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hx.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hx.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posDown = yeeLayout.getHxCircuitElement (pos, LayoutDirection::DOWN);
          GridCoordinate3D posUp = yeeLayout.getHxCircuitElement (pos, LayoutDirection::UP);
          GridCoordinate3D posBack = yeeLayout.getHxCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posFront = yeeLayout.getHxCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valHx = Hx.getFieldPointValue (pos);
          FieldPointValue* valMu = Mu.getFieldPointValue (pos);

          FieldPointValue* valEy1 = Ey.getFieldPointValue (posFront);
          FieldPointValue* valEy2 = Ey.getFieldPointValue (posBack);

          FieldPointValue* valEz1 = Ez.getFieldPointValue (posDown);
          FieldPointValue* valEz2 = Ez.getFieldPointValue (posUp);

          FieldValue val = calculateHx_3D (valHx->getPrevValue (),
                                           valEy1->getPrevValue (),
                                           valEy2->getPrevValue (),
                                           valEz1->getPrevValue (),
                                           valEz2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valMu->getCurValue ());

          FieldPointValue* tmp = Hx.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

    for (int i = 0; i < Hy.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hy.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hy.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getEyCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getEyCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posBack = yeeLayout.getEyCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posFront = yeeLayout.getEyCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valHy = Hy.getFieldPointValue (pos);
          FieldPointValue* valMu = Mu.getFieldPointValue (pos);

          FieldPointValue* valEz1 = Ez.getFieldPointValue (posRight);
          FieldPointValue* valEz2 = Ez.getFieldPointValue (posLeft);

          FieldPointValue* valEx1 = Ex.getFieldPointValue (posBack);
          FieldPointValue* valEx2 = Ex.getFieldPointValue (posFront);

          FieldValue val = calculateHy_3D (valHy->getPrevValue (),
                                           valEz1->getPrevValue (),
                                           valEz2->getPrevValue (),
                                           valEx1->getPrevValue (),
                                           valEx2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valMu->getCurValue ());

          FieldPointValue* tmp = Hy.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

    for (int i = 0; i < Hz.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Hz.getSize ().getY (); ++j)
      {
        for (int k = 0; k < Hz.getSize ().getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getEzCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getEzCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posDown = yeeLayout.getEzCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posUp = yeeLayout.getEzCircuitElement (pos, LayoutDirection::FRONT);

          FieldPointValue* valHz = Hz.getFieldPointValue (pos);
          FieldPointValue* valMu = Mu.getFieldPointValue (pos);

          FieldPointValue* valEx1 = Ex.getFieldPointValue (posUp);
          FieldPointValue* valEx2 = Ex.getFieldPointValue (posDown);

          FieldPointValue* valEy1 = Ey.getFieldPointValue (posLeft);
          FieldPointValue* valEy2 = Ey.getFieldPointValue (posRight);

          FieldValue val = calculateHz_3D (valHz->getPrevValue (),
                                           valEx1->getPrevValue (),
                                           valEx2->getPrevValue (),
                                           valEy1->getPrevValue (),
                                           valEy2->getPrevValue (),
                                           gridTimeStep,
                                           gridStep,
                                           valMu->getCurValue ());

          FieldPointValue* tmp = Hz.getFieldPointValue (pos);
          tmp->setCurValue (val);
        }
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
    Hz.nextTimeStep ();
  }

#if defined (PARALLEL_GRID)
  if (process == 0)
#endif
  {
    // BMPDumper<GridCoordinate2D> dumper;
    // dumper.init (totalStep, CURRENT);
    // dumper.dumpGrid (Ez);
  }
//#endif /* !CUDA_ENABLED */
}

void
Scheme3D::initScheme (FieldValue wLength, FieldValue step)
{
  waveLength = wLength;
  stepWaveLength = step;
  frequency = PhConst.SpeedOfLight / waveLength;

  gridStep = waveLength / stepWaveLength;
  gridTimeStep = gridStep / (2 * PhConst.SpeedOfLight);
}

#if defined (PARALLEL_GRID)
void
Scheme3D::initProcess (int rank)
{
  process = rank;
}
#endif

void
Scheme3D::initGrids ()
{
  FieldValue eps0 = PhConst.Eps0;
  FieldValue mu0 = PhConst.Mu0;

  for (int i = 0; i < Ez.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ez.getSize ().getY (); ++j)
    {
      for (int k = 0; j < Ez.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* eps = new FieldPointValue (1 * eps0, 1*eps0, 1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0, 1*mu0);

        FieldPointValue* valEz = new FieldPointValue (0, 0, 0);
        FieldPointValue* valHx = new FieldPointValue (0, 0, 0);
        FieldPointValue* valHy = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* eps = new FieldPointValue (1*eps0, 1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0);

        FieldPointValue* valEz = new FieldPointValue (0, 0);
        FieldPointValue* valHx = new FieldPointValue (0, 0);
        FieldPointValue* valHy = new FieldPointValue (0, 0);
#else
        FieldPointValue* eps = new FieldPointValue (1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0);

        FieldPointValue* valEz = new FieldPointValue (0);
        FieldPointValue* valHx = new FieldPointValue (0);
        FieldPointValue* valHy = new FieldPointValue (0);
#endif

        GridCoordinate3D pos (i, j, k);

        Eps.setFieldPointValue (eps, pos);
        Mu.setFieldPointValue (mu, pos);

        Ez.setFieldPointValue (valEz, pos);
        Hx.setFieldPointValue (valHx, pos);
        Hy.setFieldPointValue (valHy, pos);
      }
    }
  }

#if defined (PARALLEL_GRID)
  MPI_Barrier (MPI_COMM_WORLD);
#endif

#if defined (PARALLEL_GRID)
  Eps.share ();
  Mu.share ();
#endif
}

#endif /* GRID_2D */

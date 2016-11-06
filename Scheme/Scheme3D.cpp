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

extern PhysicsConst PhConst;

void
Scheme3D::performSteps (int dumpRes)
{
#if defined (CUDA_ENABLED)
  CudaExitStatus status;

  cudaExecute3DSteps (&status, yeeLayout, gridTimeStep, gridStep, Ex, Ey, Ez, Hx, Hy, Hz, Eps, Mu, totalStep, process);

  ASSERT (status == CUDA_OK);

  if (dumpRes)
  {
    BMPDumper<GridCoordinate3D> dumper;
    dumper.init (totalStep, ALL, process, "3D-TMz-in-time");
    dumper.dumpGrid (Ez);
  }
#else /* CUDA_ENABLED */

  GridCoordinate3D EzSize = Ez.getSize ();

  for (int t = 0; t < totalStep; ++t)
  {
    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getSize ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getSize ());

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getSize ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getSize ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getSize ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getSize ());

    //std::cout << t << " " << EzStart.getY () << " " << EzEnd.getY () << std::endl;

    for (int i = ExStart.getX (); i < ExEnd.getX (); ++i)
    {
      for (int j = ExStart.getY (); j < ExEnd.getY (); ++j)
      {
        for (int k = ExStart.getZ (); k < ExEnd.getZ (); ++k)
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

    for (int i = EyStart.getX (); i < EyEnd.getX (); ++i)
    {
      for (int j = EyStart.getY (); j < EyEnd.getY (); ++j)
      {
        for (int k = EyStart.getZ (); k < EyEnd.getZ (); ++k)
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

    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        for (int k = EzStart.getZ (); k < EzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getEzCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getEzCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posDown = yeeLayout.getEzCircuitElement (pos, LayoutDirection::DOWN);
          GridCoordinate3D posUp = yeeLayout.getEzCircuitElement (pos, LayoutDirection::UP);

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
      for (grid_iter k = 0; k < EzSize.getZ (); ++k)
      {
        GridCoordinate3D pos (EzSize.getX () / 2, EzSize.getY () / 2, k);
        FieldPointValue* tmp = Ez.getFieldPointValue (pos);
        tmp->setCurValue (cos (t * 3.1415 / 12));
      }
    }

    Ex.nextTimeStep ();
    Ey.nextTimeStep ();
    Ez.nextTimeStep ();

    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        for (int k = HxStart.getZ (); k < HxEnd.getZ (); ++k)
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

    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        for (int k = HyStart.getZ (); k < HyEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getHyCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getHyCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posBack = yeeLayout.getHyCircuitElement (pos, LayoutDirection::BACK);
          GridCoordinate3D posFront = yeeLayout.getHyCircuitElement (pos, LayoutDirection::FRONT);

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

    for (int i = HzStart.getX (); i < HzEnd.getX (); ++i)
    {
      for (int j = HzStart.getY (); j < HzEnd.getY (); ++j)
      {
        for (int k = HzStart.getZ (); k < HzEnd.getZ (); ++k)
        {
          GridCoordinate3D pos (i, j, k);
          GridCoordinate3D posLeft = yeeLayout.getHzCircuitElement (pos, LayoutDirection::LEFT);
          GridCoordinate3D posRight = yeeLayout.getHzCircuitElement (pos, LayoutDirection::RIGHT);
          GridCoordinate3D posDown = yeeLayout.getHzCircuitElement (pos, LayoutDirection::DOWN);
          GridCoordinate3D posUp = yeeLayout.getHzCircuitElement (pos, LayoutDirection::UP);

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

  if (dumpRes)
  {
    BMPDumper<GridCoordinate3D> dumper;
    dumper.init (totalStep, ALL, process, "3D-TMz-in-time");
    dumper.dumpGrid (Ez);
  }
#endif /* !CUDA_ENABLED */
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

  for (int i = 0; i < Eps.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Eps.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Eps.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* eps = new FieldPointValue (1 * eps0, 1*eps0, 1*eps0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* eps = new FieldPointValue (1*eps0, 1*eps0);
#else
        FieldPointValue* eps = new FieldPointValue (1*eps0);
#endif

        GridCoordinate3D pos (i, j, k);

        Eps.setFieldPointValue (eps, pos);
      }
    }
  }

  for (int i = 0; i < Mu.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Mu.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Mu.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0, 1*mu0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0);
#else
        FieldPointValue* mu = new FieldPointValue (1*eps0);
#endif

        GridCoordinate3D pos (i, j, k);

        Mu.setFieldPointValue (mu, pos);
      }
    }
  }

  for (int i = 0; i < Ex.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ex.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ex.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Ex.setFieldPointValue (val, pos);
      }
    }
  }

  for (int i = 0; i < Ey.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ey.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ey.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Ey.setFieldPointValue (val, pos);
      }
    }
  }

  for (int i = 0; i < Ez.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ez.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Ez.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Ez.setFieldPointValue (val, pos);
      }
    }
  }

  for (int i = 0; i < Hx.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hx.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hx.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Hx.setFieldPointValue (val, pos);
      }
    }
  }

  for (int i = 0; i < Hy.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hy.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hy.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Hy.setFieldPointValue (val, pos);
      }
    }
  }

  for (int i = 0; i < Hz.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Hz.getSize ().getY (); ++j)
    {
      for (int k = 0; k < Hz.getSize ().getZ (); ++k)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0, 0.0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (0.0, 0.0);
#else
        FieldPointValue* val = new FieldPointValue (0.0);
#endif

        GridCoordinate3D pos (i, j, k);

        Hz.setFieldPointValue (val, pos);
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

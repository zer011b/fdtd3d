#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "Kernels.h"
#include "PhysicsConst.h"
#include "SchemeTMz.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#ifdef GRID_2D

extern PhysicsConst PhConst;

void
SchemeTMz::performSteps ()
{
#if defined (CUDA_ENABLED)

  time_step t = 0;

#ifdef PARALLEL_GRID
  ParallelGridCoordinate bufSize = Ez.getBufferSize ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  time_step tStep = bufSize.getX ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  time_step tStep = bufSize.getY ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  time_step tStep = bufSize.getZ ();
#endif
#else
  time_step tStep = totalStep;
#endif

  while (t < totalStep)
  {
    CudaExitStatus exitStatus;

    cudaExecute2DTMzSteps (&exitStatus, yeeLayout, gridTimeStep, gridStep, Ez, Hx, Hy, Eps, Mu, 0, tStep, process);

    ASSERT (exitStatus == CUDA_OK);

#if defined (PARALLEL_GRID)
    Ez.share ();
    Hx.share ();
    Hy.share ();
#endif /* PARALLEL_GRID */

    t += tStep;
  }

  BMPDumper<GridCoordinate2D> dumper;
  dumper.init (totalStep, ALL, process);
  dumper.dumpGrid (Ez);

#else /* CUDA_ENABLED */

  GridCoordinate2D EzSize = Ez.getSize ();

  for (int t = 0; t < totalStep; ++t)
  {
    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());

    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        GridCoordinate2D pos1 (i, j);
        GridCoordinate2D pos2 (i - 1, j);
        GridCoordinate2D pos3 (i, j - 1);

        FieldPointValue* valEz = Ez.getFieldPointValue (pos1);
        FieldPointValue* valEps = Eps.getFieldPointValue (pos1);

        FieldPointValue* valHx1 = Hx.getFieldPointValue (pos3);
        FieldPointValue* valHx2 = Hx.getFieldPointValue (pos1);

        FieldPointValue* valHy1 = Hy.getFieldPointValue (pos1);
        FieldPointValue* valHy2 = Hy.getFieldPointValue (pos2);

        FieldValue val = calculateEz_2D_TMz (valEz->getPrevValue (),
                                             valHx1->getPrevValue (),
                                             valHx2->getPrevValue (),
                                             valHy1->getPrevValue (),
                                             valHy2->getPrevValue (),
                                             gridTimeStep,
                                             gridStep,
                                             valEps->getCurValue ());

        FieldPointValue* tmp = Ez.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

#if defined (PARALLEL_GRID)
    if (process == 0)
#endif
    {
      GridCoordinate2D pos (EzSize.getX () / 2, EzSize.getY () / 2);
      FieldPointValue* tmp = Ez.getFieldPointValue (pos);
      tmp->setCurValue (cos (t * 3.1415 / 12));
    }

    Ez.nextTimeStep ();

    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        GridCoordinate2D pos1 (i, j);
        GridCoordinate2D pos2 (i, j + 1);

        FieldPointValue* valHx = Hx.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos1);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos2);

        FieldValue val = calculateHx_2D_TMz (valHx->getPrevValue (),
                                             valEz1->getPrevValue (),
                                             valEz2->getPrevValue (),
                                             gridTimeStep,
                                             gridStep,
                                             valMu->getCurValue ());

        FieldPointValue* tmp = Hx.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        GridCoordinate2D pos1 (i, j);
        GridCoordinate2D pos2 (i + 1, j);

        FieldPointValue* valHy = Hy.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos2);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos1);

        FieldValue val = calculateHy_2D_TMz (valHy->getPrevValue (),
                                             valEz1->getPrevValue (),
                                             valEz2->getPrevValue (),
                                             gridTimeStep,
                                             gridStep,
                                             valMu->getCurValue ());

        FieldPointValue* tmp = Hy.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
  }

  BMPDumper<GridCoordinate2D> dumper;
  dumper.init (totalStep, CURRENT, process);
  dumper.dumpGrid (Ez);
#endif /* !CUDA_ENABLED */
}

void
SchemeTMz::initScheme (FieldValue wLength, FieldValue step)
{
  waveLength = wLength;
  stepWaveLength = step;
  frequency = PhConst.SpeedOfLight / waveLength;

  gridStep = waveLength / stepWaveLength;
  gridTimeStep = gridStep / (2 * PhConst.SpeedOfLight);
}

#if defined (PARALLEL_GRID)
void
SchemeTMz::initProcess (int rank)
{
  process = rank;
}
#endif

void
SchemeTMz::initGrids ()
{
  FieldValue eps0 = PhConst.Eps0;
  FieldValue mu0 = PhConst.Mu0;

  for (int i = 0; i < Ez.getSize ().getX (); ++i)
  {
    for (int j = 0; j < Ez.getSize ().getY (); ++j)
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

      GridCoordinate2D pos (i, j);

      Eps.setFieldPointValue (eps, pos);
      Mu.setFieldPointValue (mu, pos);

      Ez.setFieldPointValue (valEz, pos);
      Hx.setFieldPointValue (valHx, pos);
      Hy.setFieldPointValue (valHy, pos);
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

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
SchemeTMz::performEzSteps (time_step t, GridCoordinate3D EzStart, GridCoordinate3D EzEnd)
{
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
}

void
SchemeTMz::performHxSteps (time_step t, GridCoordinate3D HxStart, GridCoordinate3D HxEnd)
{
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
}

void
SchemeTMz::performHySteps (time_step t, GridCoordinate3D HyStart, GridCoordinate3D HyEnd)
{
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
}

void
SchemeTMz::performNSteps (time_step startStep, time_step numberTimeSteps, int dumpRes)
{
  GridCoordinate2D EzSize = Ez.getSize ();

  time_step stepLimit = startStep + numberTimeSteps;

  for (int t = startStep; t < stepLimit; ++t)
  {
    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());

    performEzSteps (t, EzStart, EzEnd);

#if defined (PARALLEL_GRID)
    if (process == 0)
#endif
    {
      GridCoordinate2D pos (EzSize.getX () / 2, EzSize.getY () / 2);
      FieldPointValue* tmp = Ez.getFieldPointValue (pos);
      tmp->setCurValue (cos (t * 3.1415 / 12));
    }

    Ez.nextTimeStep ();

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (stepLimit, CURRENT, process, "2D-TMz-in-time");
    dumper.dumpGrid (Ez);
  }
}

void
SchemeTMz::performAmplitudeSteps (time_step startStep, int dumpRes)
{
  int is_stable_state = 0;

  GridCoordinate2D EzSize = Ez.getSize ();

  time_step t = startStep;

  while (is_stable_state == 0 && t < amplitudeStepLimit)
  {
    FieldValue maxAccuracy = -1;

    is_stable_state = 1;

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());

    performEzSteps (t, EzStart, EzEnd);

    for (int i = EzStart.getX (); i < EzEnd.getX (); ++i)
    {
      for (int j = EzStart.getY (); j < EzEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        FieldPointValue* tmp = Ez.getFieldPointValue (pos);
        FieldPointValue* tmpAmp = EzAmplitude.getFieldPointValue (pos);

        if (updateAmplitude (tmp, tmpAmp, &maxAccuracy) == 0)
        {
          is_stable_state = 0;
        }
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

    performHxSteps (t, HxStart, HxEnd);
    performHySteps (t, HyStart, HyEnd);

    for (int i = HxStart.getX (); i < HxEnd.getX (); ++i)
    {
      for (int j = HxStart.getY (); j < HxEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        FieldPointValue* tmp = Hx.getFieldPointValue (pos);
        FieldPointValue* tmpAmp = HxAmplitude.getFieldPointValue (pos);

        if (updateAmplitude (tmp, tmpAmp, &maxAccuracy) == 0)
        {
          is_stable_state = 0;
        }
      }
    }

    for (int i = HyStart.getX (); i < HyEnd.getX (); ++i)
    {
      for (int j = HyStart.getY (); j < HyEnd.getY (); ++j)
      {
        GridCoordinate2D pos (i, j);

        FieldPointValue* tmp = Hy.getFieldPointValue (pos);
        FieldPointValue* tmpAmp = HyAmplitude.getFieldPointValue (pos);

        if (updateAmplitude (tmp, tmpAmp, &maxAccuracy) == 0)
        {
          is_stable_state = 0;
        }
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();

    ++t;

    if (maxAccuracy < 0)
    {
      is_stable_state = 0;
    }

#if defined (PARALLEL_GRID)
    for (int rank = 0; rank < Ez.getTotalProcCount (); ++rank)
    {
      if (process == rank)
      {
        for (int rankDest = 0; rankDest < Ez.getTotalProcCount (); ++rankDest)
        {
          if (rankDest != rank)
          {
            int retCode = MPI_Send (&is_stable_state, 1, MPI_INT, 0, process, MPI_COMM_WORLD);

            ASSERT (retCode == MPI_SUCCESS);
          }
        }
      }
      else
      {
        MPI_Status status;

        int is_other_stable_state = 0;

        int retCode = MPI_Recv (&is_other_stable_state, 1, MPI_INT, rank, rank, MPI_COMM_WORLD, &status);

        ASSERT (retCode == MPI_SUCCESS);

        if (!is_other_stable_state)
        {
          is_stable_state = 0;
        }
      }

      MPI_Barrier (MPI_COMM_WORLD);
    }
#endif

#if PRINT_MESSAGE
    printf ("%d amplitude calculation step: max accuracy %f. \n", t, maxAccuracy);
#endif /* PRINT_MESSAGE */
  }

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (t, CURRENT, process, "2D-TMz-amplitude");
    dumper.dumpGrid (Ez);
  }

  if (is_stable_state == 0)
  {
    ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.\n");
  }
}

int
SchemeTMz::updateAmplitude (FieldPointValue *fieldValue, FieldPointValue *amplitudeValue, FieldValue *maxAccuracy)
{
  int is_stable_state = 1;

  FieldValue val = fieldValue->getCurValue ();
  FieldValue valAmp = amplitudeValue->getCurValue ();

  val = val >= 0 ? val : -val;

  if (val > valAmp)
  {
    FieldValue accuracy = val - valAmp;
    accuracy /= valAmp;

    if (accuracy > PhysicsConst::accuracy)
    {
      is_stable_state = 0;

      amplitudeValue->setCurValue (val);
    }

    if (accuracy > *maxAccuracy)
    {
      *maxAccuracy = accuracy;
    }
  }

  return is_stable_state;
}

void
SchemeTMz::performSteps (int dumpRes)
{
#if defined (CUDA_ENABLED)
  CudaExitStatus status;

  cudaExecute2DTMzSteps (&status, yeeLayout, gridTimeStep, gridStep, Ez, Hx, Hy, Eps, Mu, totalStep, process);

  ASSERT (status == CUDA_OK);

  if (dumpRes)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (totalStep, ALL, process, "2D-TMz-in-time");
    dumper.dumpGrid (Ez);
  }
#else /* CUDA_ENABLED */

  performNSteps (0, totalStep, dumpRes);

  if (calculateAmplitude)
  {
    performAmplitudeSteps (totalStep, dumpRes);
  }

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
      FieldPointValue* valEzAmp;
      FieldPointValue* valHxAmp;
      FieldPointValue* valHyAmp;

#if defined (TWO_TIME_STEPS)
      FieldPointValue* eps = new FieldPointValue (1 * eps0, 1*eps0, 1*eps0);
      FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0, 1*mu0);

      FieldPointValue* valEz = new FieldPointValue (0, 0, 0);
      FieldPointValue* valHx = new FieldPointValue (0, 0, 0);
      FieldPointValue* valHy = new FieldPointValue (0, 0, 0);

      if (calculateAmplitude)
      {
        valEzAmp = new FieldPointValue (0, 0, 0);
        valHxAmp = new FieldPointValue (0, 0, 0);
        valHyAmp = new FieldPointValue (0, 0, 0);
      }
#elif defined (ONE_TIME_STEP)
      FieldPointValue* eps = new FieldPointValue (1*eps0, 1*eps0);
      FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0);

      FieldPointValue* valEz = new FieldPointValue (0, 0);
      FieldPointValue* valHx = new FieldPointValue (0, 0);
      FieldPointValue* valHy = new FieldPointValue (0, 0);

      if (calculateAmplitude)
      {
        valEzAmp = new FieldPointValue (0, 0);
        valHxAmp = new FieldPointValue (0, 0);
        valHyAmp = new FieldPointValue (0, 0);
      }
#else
      FieldPointValue* eps = new FieldPointValue (1*eps0);
      FieldPointValue* mu = new FieldPointValue (1*mu0);

      FieldPointValue* valEz = new FieldPointValue (0);
      FieldPointValue* valHx = new FieldPointValue (0);
      FieldPointValue* valHy = new FieldPointValue (0);

      if (calculateAmplitude)
      {
        valEzAmp = new FieldPointValue (0);
        valHxAmp = new FieldPointValue (0);
        valHyAmp = new FieldPointValue (0);
      }
#endif

      GridCoordinate2D pos (i, j);

      Eps.setFieldPointValue (eps, pos);
      Mu.setFieldPointValue (mu, pos);

      Ez.setFieldPointValue (valEz, pos);
      Hx.setFieldPointValue (valHx, pos);
      Hy.setFieldPointValue (valHy, pos);

      if (calculateAmplitude)
      {
        EzAmplitude.setFieldPointValue (valEzAmp, pos);
        HxAmplitude.setFieldPointValue (valHxAmp, pos);
        HyAmplitude.setFieldPointValue (valHyAmp, pos);
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

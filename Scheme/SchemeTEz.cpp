#include "SchemeTEz.h"
#include "PhysicsConst.h"
#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include <mpi.h>
#include <cmath>

extern PhysicsConst PhConst;

void
SchemeTEz::performStep ()
{
  for (uint32_t t = 0; t < totalStep; ++t)
  {
    for (int i = 1; i < Ez.getSize ().getX (); ++i)
    {
      for (int j = 1; j < Ez.getSize ().getY (); ++j)
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

        FieldValue val = valEz->getPrevValue () + (gridTimeStep / (valEps->getCurValue () * gridStep)) *
          (valHx1->getPrevValue () - valHx2->getPrevValue () + valHy1->getPrevValue () - valHy2->getPrevValue ());

        FieldPointValue* tmp = Ez.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    if (process == 0)
    {
      GridCoordinate2D pos (Ez.getSize ().getX () / 2, Ez.getSize ().getY () / 2);
      FieldPointValue* tmp = Ez.getFieldPointValue (pos);
      tmp->setCurValue (cos (t * 3.1415 / 12));
    }

    Ez.nextTimeStep ();
    Ez.Share ();

    for (int i = 1; i < Ez.getSize ().getX (); ++i)
    {
      for (int j = 0; j < Ez.getSize ().getY () - 1; ++j)
      {
        GridCoordinate2D pos1 (i, j);
        GridCoordinate2D pos2 (i, j + 1);

        FieldPointValue* valHx = Hx.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos1);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos2);

        FieldValue val = valHx->getPrevValue () + (gridTimeStep / (valMu->getCurValue () * gridStep)) *
          (valEz1->getPrevValue () - valEz2->getPrevValue ());

        FieldPointValue* tmp = Hx.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    for (int i = 0; i < Ez.getSize ().getX () - 1; ++i)
    {
      for (int j = 1; j < Ez.getSize ().getY (); ++j)
      {
        GridCoordinate2D pos1 (i, j);
        GridCoordinate2D pos2 (i + 1, j);

        FieldPointValue* valHy = Hy.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos2);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos1);

        FieldValue val = valHy->getPrevValue () + (gridTimeStep / (valMu->getCurValue () * gridStep)) *
          (valEz1->getPrevValue () - valEz2->getPrevValue ());

        FieldPointValue* tmp = Hy.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
  }

  if (process == 0)
  {
    BMPDumper<GridCoordinate2D> dumper;
    dumper.init (1000, CURRENT);
    dumper.dumpGrid (Ez);
  }
}

void
SchemeTEz::initScheme (FieldValue wLength, FieldValue step)
{
  waveLength = wLength;
  stepWaveLength = step;
  frequency = PhConst.SpeedOfLight / waveLength;

  gridStep = waveLength / stepWaveLength;
  gridTimeStep = gridStep / (2 * PhConst.SpeedOfLight);
}

void
SchemeTEz::initGrids (int rank)
{
  process = rank;

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
  Eps.Share ();
  Mu.Share ();
#endif
}

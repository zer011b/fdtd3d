#include <iostream>

#include <mpi.h>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#else
#include "Grid.h"
#endif

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

int main (int argc, char** argv)
{
#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#if PRINT_MESSAGE
  printf ("Start process %d of %d\n", rank, numProcs);
#endif
#endif

  GridCoordinate3D overallSize (100);
  //GridCoordinate size (100, 100);
  GridCoordinate3D bufferLeft (10);
  GridCoordinate3D bufferRight (10);

#if defined (PARALLEL_GRID)
  ParallelGrid Eps (overallSize, bufferLeft, bufferRight, rank, numProcs, 0);
  ParallelGrid Mu (overallSize, bufferLeft, bufferRight, rank, numProcs, 0);

  ParallelGrid Ez (overallSize, bufferLeft, bufferRight, rank, numProcs, 0);
  ParallelGrid Hx (overallSize, bufferLeft, bufferRight, rank, numProcs, 0);
  ParallelGrid Hy (overallSize, bufferLeft, bufferRight, rank, numProcs, 0);
#else
  Grid<GridCoordinate3D> Eps (overallSize, 0);
  Grid<GridCoordinate3D> Mu (overallSize, 0);

  Grid<GridCoordinate3D> Ez (overallSize, 0);
  Grid<GridCoordinate3D> Hx (overallSize, 0);
  Grid<GridCoordinate3D> Hy (overallSize, 0);
#endif

  GridCoordinate3D sizeTotal = Eps.getSize ();

  FieldValue lambda = 0.000003;
  FieldValue stepLambda = 20;
  FieldValue delta = lambda / stepLambda;
  FieldValue cc = 2.99792458e+8;
  FieldValue dt = delta / (2.0 * cc);
  FieldValue eps0 = 0.0000000000088541878176203892;
  FieldValue mu0 = 0.0000012566370614359173;

//#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (int i = 0; i < sizeTotal.getX (); ++i)
  {
//#endif
//#if defined (GRID_2D) || defined (GRID_3D)
    for (int j = 0; j < sizeTotal.getY (); ++j)
    {
//#endif
//#if defined (GRID_3D)
      for (int k = 0; k < sizeTotal.getZ (); ++k)
      {
//#endif

#if defined (TWO_TIME_STEPS)
        FieldPointValue* eps = new FieldPointValue (1*eps0, 1*eps0, 1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0, 1*mu0);

        FieldPointValue* valE = new FieldPointValue (0, 0, 0);
        FieldPointValue* valHx = new FieldPointValue (0, 0, 0);
        FieldPointValue* valHy = new FieldPointValue (0, 0, 0);
#elif defined (ONE_TIME_STEP)
        FieldPointValue* eps = new FieldPointValue (1*eps0, 1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0, 1*mu0);

        FieldPointValue* valE = new FieldPointValue (0, 0);
        FieldPointValue* valHx = new FieldPointValue (0, 0);
        FieldPointValue* valHy = new FieldPointValue (0, 0);
#else
        FieldPointValue* eps = new FieldPointValue (1*eps0);
        FieldPointValue* mu = new FieldPointValue (1*mu0);

        FieldPointValue* valE = new FieldPointValue (0);
        FieldPointValue* valHx = new FieldPointValue (0);
        FieldPointValue* valHy = new FieldPointValue (0);
#endif

#if defined (GRID_1D)
        GridCoordinate1D pos (i);
#endif
#if defined (GRID_2D)
        GridCoordinate2D pos (i, j);
#endif
#if defined (GRID_3D)
        GridCoordinate3D pos (i, j, k);
#endif

        Eps.setFieldPointValue(eps, pos);
        Mu.setFieldPointValue(mu, pos);

        Ez.setFieldPointValue(valE, pos);
        Hx.setFieldPointValue(valHx, pos);
        Hy.setFieldPointValue(valHy, pos);
//#if defined (GRID_3D)
      }
//#endif
//#if defined (GRID_2D) || defined (GRID_3D)
    }
//#endif
//#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  }
//#endif

#if defined (PARALLEL_GRID)
  MPI_Barrier (MPI_COMM_WORLD);
#endif

#if defined (PARALLEL_GRID)
  Eps.Share ();
  Mu.Share ();
#endif

/*  for (int t = 0; t < 100; ++t)
  {
    //printf ("Step %d #%d.\n", t, rank);

    for (int i = 1; i < sizeTotal.getX (); ++i)
    {
      for (int j = 1; j < sizeTotal.getY (); ++j)
      {
        GridCoordinate pos1 (i, j);
        GridCoordinate pos2 (i - 1, j);
        GridCoordinate pos3 (i, j - 1);

        FieldPointValue* valEz = Ez.getFieldPointValue (pos1);
        FieldPointValue* valEps = Eps.getFieldPointValue (pos1);

        FieldPointValue* valHx1 = Hx.getFieldPointValue (pos3);
        FieldPointValue* valHx2 = Hx.getFieldPointValue (pos1);

        FieldPointValue* valHy1 = Hy.getFieldPointValue (pos1);
        FieldPointValue* valHy2 = Hy.getFieldPointValue (pos2);

        FieldValue val = valEz->getPrevValue () + (dt / (valEps->getCurValue () * delta)) *
          (valHx1->getPrevValue () - valHx2->getPrevValue () + valHy1->getPrevValue () - valHy2->getPrevValue ());

        FieldPointValue* tmp = Ez.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    if (rank == 0)
    {
      GridCoordinate pos (sizeTotal.getX () / 2, sizeTotal.getY () / 2);
      FieldPointValue* tmp = Ez.getFieldPointValue (pos);
      tmp->setCurValue (cos (t * 3.1415 / 12));
    }

    Ez.nextTimeStep ();
    Ez.Share ();

    for (int i = 1; i < sizeTotal.getX (); ++i)
    {
      for (int j = 0; j < sizeTotal.getY () - 1; ++j)
      {
        GridCoordinate pos1 (i, j);
        GridCoordinate pos2 (i, j + 1);

        FieldPointValue* valHx = Hx.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos1);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos2);

        FieldValue val = valHx->getPrevValue () + (dt / (valMu->getCurValue () * delta)) *
          (valEz1->getPrevValue () - valEz2->getPrevValue ());

        FieldPointValue* tmp = Hx.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    for (int i = 0; i < sizeTotal.getX () - 1; ++i)
    {
      for (int j = 1; j < sizeTotal.getY (); ++j)
      {
        GridCoordinate pos1 (i, j);
        GridCoordinate pos2 (i + 1, j);

        FieldPointValue* valHy = Hy.getFieldPointValue (pos1);
        FieldPointValue* valMu = Mu.getFieldPointValue (pos1);

        FieldPointValue* valEz1 = Ez.getFieldPointValue (pos2);
        FieldPointValue* valEz2 = Ez.getFieldPointValue (pos1);

        FieldValue val = valHy->getPrevValue () + (dt / (valMu->getCurValue () * delta)) *
          (valEz1->getPrevValue () - valEz2->getPrevValue ());

        FieldPointValue* tmp = Hy.getFieldPointValue (pos1);
        tmp->setCurValue (val);
      }
    }

    Hx.nextTimeStep ();
    Hy.nextTimeStep ();
  }
*/
  /*for (int t = 2; t < 1000; ++t)
  {
    grid.nextTimeStep ();

    for (int i = 0; i < sizeTotal.getX (); ++i)
    {
      for (int j = 0; j < sizeTotal.getY (); ++j)
      {
        GridCoordinate pos (i, j);
        FieldPointValue* val = grid.getFieldPointValue (pos);

        if ((val->getPrevValue () != t - 1 ||
            val->getPrevPrevValue () != t - 2) && t > 5)
        {
          printf("%f %d\n", val->getPrevValue (), t - 1);
          return 1;
        }
        FieldPointValue* tmp = grid.getFieldPointValue (pos);
        tmp->setCurValue (t);
      }
    }
  }*/

  /*GridCoordinate size (3, 3);
  Grid grid (size);

  FieldPointValue* val1 = new FieldPointValue (0, 100, 100);
  GridCoordinate pos1 (0, 0);
  grid.setFieldPointValue(val1, pos1);

  FieldPointValue* val2 = new FieldPointValue (0, 0, 0);
  GridCoordinate pos2 (1, 0);
  grid.setFieldPointValue(val2, pos2);

  FieldPointValue* val3 = new FieldPointValue (100, 25, 25);
  GridCoordinate pos3 (2, 0);
  grid.setFieldPointValue(val3, pos3);

  FieldPointValue* val4 = new FieldPointValue (0, 100, 100);
  GridCoordinate pos4 (0, 1);
  grid.setFieldPointValue(val4, pos4);

  FieldPointValue* val5 = new FieldPointValue (100, 100, 15);
  GridCoordinate pos5 (1, 1);
  grid.setFieldPointValue(val5, pos5);

  FieldPointValue* val6 = new FieldPointValue (100, 10, 100);
  GridCoordinate pos6 (2, 1);
  grid.setFieldPointValue(val6, pos6);

  FieldPointValue* val7 = new FieldPointValue (0, 75, 75);
  GridCoordinate pos7 (0, 2);
  grid.setFieldPointValue(val7, pos7);

  FieldPointValue* val8 = new FieldPointValue (0, 0, 0);
  GridCoordinate pos8 (1, 2);
  grid.setFieldPointValue(val8, pos8);

  FieldPointValue* val9 = new FieldPointValue (100, 0, 100);
  GridCoordinate pos9 (2, 2);
  grid.setFieldPointValue(val9, pos9);



  DATDumper dumper;
  dumper.init (1, ALL);
  dumper.dumpGrid (grid);*/

  /*DATLoader loader;
  //FieldPointValue val10 (100, 100, 100);
  //loader.setMaxValuePos (val10);
  //FieldPointValue val11 (0, 0, 0);
  //loader.setMaxValueNeg (val11);
  loader.init (1, ALL);
  loader.loadGrid (grid);

  GridCoordinate pos5 (2, 1);
  const FieldPointValue* val_1 = grid.getFieldPointValue (pos5);
  std::cout << val_1->getCurValue () << ", " <<
    val_1->getPrevValue() << ", " << val_1->getPrevPrevValue() << std::endl;*/


  // if (rank == 0)
  // {
  //   GridCoordinate pos (1, 1, 1);
  //   const FieldPointValue* val = Ez.getFieldPointValue (pos);
  //   printf ("%f %f %f\n", val->getCurValue (), val->getPrevValue(),
  //     val->getPrevPrevValue());
  //
  //   BMPDumper dumper;
  //   dumper.init (1000, CURRENT);
  //   dumper.dumpGrid (Ez);
  //   /*dumper.dumpGrid (Hx);
  //   dumper.dumpGrid (Hy);*/
  // }

#if defined (PARALLEL_GRID)
#if PRINT_MESSAGE
  printf ("Main process %d.\n", rank);
#endif

  MPI_Finalize();
#endif

  return 0;
}

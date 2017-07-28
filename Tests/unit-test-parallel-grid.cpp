/*
 * Unit test for ParallelGrid
 *
 * Data for each computational node of ParallelGrid is assigned equal to the id of the corresponding computational node
 * (i.e. process 0 has its data filled with 0):
 *   id for real part of current step values, id * 1000 for imaginary part of current step values
 *   id * 16 for real part of current step values, id * 16 * 1000 for imaginary part of current step values
 *   id * 256 for real part of current step values, id * 256 * 1000 for imaginary part of current step values
 *
 * Then all data is gather on all the nodes and checked for consistency.
 *
 * Number of computational nodes is set to be divider of grid size for all dimensions.
 */

#include <iostream>

#include "Assert.h"

#ifdef PARALLEL_GRID

#include "ParallelGrid.h"
#include "ParallelYeeGridLayout.h"
#include <mpi.h>

#ifdef CXX11_ENABLED
#else /* CXX11_ENABLED */
#include "cstdlib"
#endif /* !CXX11_ENABLED */

const FPValue imagMult = 1000;
const FPValue prevMult = 16;
const FPValue prevPrevMult = prevMult * prevMult;

int main (int argc, char** argv)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter gridSizeX = 32;
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter gridSizeY = 32;
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
  grid_iter gridSizeZ = 32;
#endif /* GRID_3D */

  int bufSize = 2;

  MPI_Init (&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  DPRINTF ("X: PID %d of %d, grid size x: %lu\n", rank, numProcs, gridSizeX);
  ASSERT (gridSizeX % numProcs == 0);
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
  DPRINTF ("Y: PID %d of %d, grid size y: %lu\n", rank, numProcs, gridSizeY);
  ASSERT (gridSizeY % numProcs == 0);
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
  DPRINTF ("Z: PID %d of %d, grid size x: %lu\n", rank, numProcs, gridSizeZ);
  ASSERT (gridSizeZ % numProcs == 0);
#endif /* GRID_3D */

  DPRINTF (LOG_LEVEL_STAGES, "Start process %d of %d\n", rank, numProcs);

#ifdef GRID_1D
  GridCoordinate3D overallSize (gridSizeX, 0, 0);
  GridCoordinate3D pmlSize (10, 0, 0);
  GridCoordinate3D tfsfSize (20, 0, 0);
  GridCoordinate3D bufferSize (bufSize, 0, 0);
#endif /* GRID_1D */

#ifdef GRID_2D
  GridCoordinate3D overallSize (gridSizeX, gridSizeY, 0);
  GridCoordinate3D pmlSize (10, 10, 0);
  GridCoordinate3D tfsfSize (20, 20, 0);
  GridCoordinate3D bufferSize (bufSize, bufSize, 0);
#endif /* GRID_2D */

#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSizeX, gridSizeY, gridSizeZ);
  GridCoordinate3D pmlSize (10, 10, 10);
  GridCoordinate3D tfsfSize (20, 20, 20);
  GridCoordinate3D bufferSize (bufSize, bufSize, bufSize);
#endif /* GRID_3D */

  ParallelGridCore parallelGridCore (rank, numProcs, overallSize);
  ParallelGrid::initializeParallelCore (&parallelGridCore);

  bool isDoubleMaterialPrecision = false;

  ParallelYeeGridLayout yeeLayout (overallSize, pmlSize, tfsfSize, PhysicsConst::Pi / 2, 0, 0, isDoubleMaterialPrecision);
  yeeLayout.Initialize (parallelGridCore);

  ParallelGrid grid (overallSize, bufferSize, 0, yeeLayout.getSizeForCurNode (), yeeLayout.getCoreSizePerNode ());

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_iter i = 0; i < grid.getSize ().getX (); ++i)
  {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_iter j = 0; j < grid.getSize ().getY (); ++j)
    {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      for (grid_iter k = 0; k < grid.getSize ().getZ (); ++k)
      {
#endif /* GRID_3D */
        FieldPointValue* val = new FieldPointValue ();

#ifdef GRID_1D
        GridCoordinate1D pos (i);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k);
#endif /* GRID_3D */

        ParallelGridCoordinate posAbs = grid.getTotalPosition (pos);

        FPValue fpval = 0;

        fpval = ParallelGrid::getParallelCore ()->getProcessId ();

#ifdef COMPLEX_FIELD_VALUES

        val->setCurValue (FieldValue (fpval, fpval * imagMult));

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        val->setPrevValue (FieldValue (fpval * prevMult, fpval * prevMult * imagMult));
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
        val->setPrevPrevValue (FieldValue (fpval * prevPrevMult, fpval * prevPrevMult * imagMult));
#endif /* TWO_TIME_STEPS */

#else /* COMPLEX_FIELD_VALUES */

        val->setCurValue (fpval);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        val->setPrevValue (fpval * prevMult);
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
        val->setPrevPrevValue (fpval * prevPrevMult);
#endif /* TWO_TIME_STEPS */

#endif /* !COMPLEX_FIELD_VALUES */

        grid.setFieldPointValue (val, pos);

#if defined (GRID_3D)
      }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  }
#endif /* GRID_1D || GRID_2D || GRID_3D */

  grid.share ();

  ParallelGridBase *gridTotal = grid.gatherFullGrid ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_iter i = 0; i < grid.getSize ().getX (); ++i)
  {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_iter j = 0; j < grid.getSize ().getY (); ++j)
    {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      for (grid_iter k = 0; k < grid.getSize ().getZ (); ++k)
      {
#endif /* GRID_3D */

#ifdef GRID_1D
        GridCoordinate1D pos (i);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k);
#endif /* GRID_3D */

        FieldPointValue *val = gridTotal->getFieldPointValue (pos);

        FPValue fpval;

#ifdef COMPLEX_FIELD_VALUES

        fpval = val->getCurValue ().real ();
        ASSERT (fpval * imagMult == val->getCurValue ().imag ());

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        ASSERT (fpval * prevMult == val->getPrevValue ().real ());
        ASSERT (fpval * prevMult * imagMult == val->getPrevValue ().imag ());
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
        ASSERT (fpval * prevPrevMult == val->getPrevPrevValue ().real ());
        ASSERT (fpval * prevPrevMult * imagMult == val->getPrevPrevValue ().imag ());
#endif /* TWO_TIME_STEPS */

#else /* COMPLEX_FIELD_VALUES */

        fpval = val->getCurValue ();

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        ASSERT (fpval * prevMult == val->getPrevValue ());
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
        ASSERT (fpval * prevPrevMult == val->getPrevPrevValue ());
#endif /* TWO_TIME_STEPS */

#endif /* !COMPLEX_FIELD_VALUES */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
        grid_coord step = gridTotal->getSize ().getX () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        int process = i / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
        grid_coord step = gridTotal->getSize ().getY () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        int process = j / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
        grid_coord step = gridTotal->getSize ().getZ () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();
        int process = k / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
        grid_coord stepX = gridTotal->getSize ().getX () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepY = gridTotal->getSize ().getY () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();

        int processI = i / stepX;
        int processJ = j / stepY;

        int process = processJ * ParallelGrid::getParallelCore ()->getNodeGridSizeX () + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
        grid_coord stepY = gridTotal->getSize ().getY () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        grid_coord stepZ = gridTotal->getSize ().getZ () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processJ = j / stepY;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeY () + processJ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
        grid_coord stepX = gridTotal->getSize ().getX () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepZ = gridTotal->getSize ().getZ () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processI = i / stepX;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeX () + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
        grid_coord stepX = gridTotal->getSize ().getX () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepY = gridTotal->getSize ().getY () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        grid_coord stepZ = gridTotal->getSize ().getZ () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processI = i / stepX;
        int processJ = j / stepY;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeXY ()
                      + processJ * ParallelGrid::getParallelCore ()->getNodeGridSizeX ()
                      + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

        FPValue fpprocess = (FPValue) process;

        ASSERT (fpprocess == fpval);

#if defined (GRID_3D)
      }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  }
#endif /* GRID_1D || GRID_2D || GRID_3D */

  delete gridTotal;

  MPI_Finalize();

  return 0;
} /* main */

#else /* PARALLEL_GRID */

int main (int argc, char** argv)
{
  ASSERT (0);

  return 0;
} /* main */

#endif /* !PARALLEL_GRID */

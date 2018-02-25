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
#include "Settings.h"

#ifdef PARALLEL_GRID

#include "ParallelGrid.h"
#include "ParallelYeeGridLayout.h"
#include <mpi.h>

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

const FPValue imagMult = 1000;
const FPValue prevMult = 16;
const FPValue prevPrevMult = prevMult * prevMult;

std::vector<ParallelGridCoordinate> neighbourSendStart (BUFFER_COUNT);
std::vector<ParallelGridCoordinate> neighbourSendEnd (BUFFER_COUNT);

void checkVal (ParallelGrid *grid, ParallelGridCoordinate pos)
{
  FieldPointValue *val = grid->getFieldPointValue (pos);
  BufferPosition dir = grid->getBufferForPosition (pos);

  if (dir == BUFFER_NONE)
  {
    return;
  }

  BufferPosition opposite = ParallelGrid::getParallelCore ()->getOppositeDirections ()[dir];
  ParallelGridCoordinate posInNeighbor = pos - grid->getRecvStart ()[opposite] + neighbourSendStart[dir];

#ifdef GRID_1D
  FPValue multiplier = posInNeighbor.get1 ();
#endif
#ifdef GRID_2D
  FPValue multiplier = posInNeighbor.get1 () * posInNeighbor.get2 ();
#endif
#ifdef GRID_3D
  FPValue multiplier = posInNeighbor.get1 () * posInNeighbor.get2 () * posInNeighbor.get3 ();
#endif

  int pidSender = ParallelGrid::getParallelCore ()->getNodeForDirection (dir);

  ParallelGridCoordinate sendStart = grid->getSendStart ()[dir];

  FieldValue cur = val->getCurValue ();
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (cur.real () == pidSender * multiplier);
  ASSERT (cur.imag () == pidSender * multiplier * imagMult);
#else
  ASSERT (cur == pidSender * multiplier);
#endif

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue prev = val->getPrevValue ();
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (prev.real () == pidSender * multiplier * prevMult);
  ASSERT (prev.imag () == pidSender * multiplier * prevMult * imagMult);
#else
  ASSERT (prev == pidSender * multiplier * prevMult);
#endif
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
  FieldValue prevPrev = val->getPrevPrevValue ();
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (prevPrev.real () == pidSender * multiplier * prevPrevMult);
  ASSERT (prevPrev.imag () == pidSender * multiplier * prevPrevMult * imagMult);
#else
  ASSERT (prevPrev == pidSender * multiplier * prevPrevMult);
#endif
#endif /* TWO_TIME_STEPS */
}

ParallelGrid * initGrid (ParallelGridCoordinate overallSize,
                         ParallelGridCoordinate bufferSize,
                         ParallelGridCoordinate sizeForCurNode,
                         bool isAbsVal)
{
  ParallelGrid *grid = new ParallelGrid (overallSize, bufferSize, 0, sizeForCurNode);

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
      {
#endif /* GRID_3D */
        FieldPointValue* val = new FieldPointValue ();

#ifdef GRID_1D
        GridCoordinate1D pos (i, CoordinateType::X);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j, CoordinateType::X, CoordinateType::Y);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
#endif /* GRID_3D */

        ParallelGridCoordinate totalPos = grid->getTotalPosition (pos);
        ParallelGridCoordinate coord;

        if (isAbsVal)
        {
          coord = totalPos;
        }
        else
        {
          coord = pos;
        }

        FPValue fpval = 0;

        fpval = ParallelGrid::getParallelCore ()->getProcessId ();
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
        fpval *= coord.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
        fpval *= coord.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        fpval *= coord.get3 ();
#endif /* GRID_3D */

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

        grid->setFieldPointValue (val, pos);

#if defined (GRID_3D)
      }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  }
#endif /* GRID_1D || GRID_2D || GRID_3D */

  return grid;
}

int main (int argc, char** argv)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord gridSizeX = 32;
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord gridSizeY = 32;
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
  grid_coord gridSizeZ = 32;
#endif /* GRID_3D */

  int bufSize = 2;

  int res = MPI_Init (&argc, &argv);

  ALWAYS_ASSERT (res == MPI_SUCCESS);

  int rank, numProcs;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  DPRINTF (LOG_LEVEL_STAGES, "X: PID %d of %d, grid size x: " COORD_MOD "\n", rank, numProcs, gridSizeX);
  ASSERT (gridSizeX % numProcs == 0);
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
  DPRINTF (LOG_LEVEL_STAGES, "Y: PID %d of %d, grid size y: " COORD_MOD "\n", rank, numProcs, gridSizeY);
  ASSERT (gridSizeY % numProcs == 0);
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
  DPRINTF (LOG_LEVEL_STAGES, "Z: PID %d of %d, grid size x: " COORD_MOD "\n", rank, numProcs, gridSizeZ);
  ASSERT (gridSizeZ % numProcs == 0);
#endif /* GRID_3D */

  DPRINTF (LOG_LEVEL_STAGES, "Start process %d of %d\n", rank, numProcs);

#ifdef GRID_1D
  GridCoordinate1D overallSize (gridSizeX, CoordinateType::X);
  GridCoordinate1D pmlSize (10, CoordinateType::X);
  GridCoordinate1D tfsfSize (20, CoordinateType::X);
  GridCoordinate1D bufferSize (bufSize, CoordinateType::X);
  GridCoordinate1D topologySize (0, CoordinateType::X);

#define SCHEME_TYPE SchemeType::Dim1_EzHy
#define ANGLES PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2
#endif /* GRID_1D */

#ifdef GRID_2D
  GridCoordinate2D overallSize (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D pmlSize (10, 10, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D tfsfSize (20, 20, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D bufferSize (bufSize, bufSize, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D topologySize (0, 0, CoordinateType::X, CoordinateType::Y);

#define SCHEME_TYPE SchemeType::Dim2_TEz
#define ANGLES PhysicsConst::Pi / 2, 0, 0
#endif /* GRID_2D */

#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D pmlSize (10, 10, 10, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D tfsfSize (20, 20, 20, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D bufferSize (bufSize, bufSize, bufSize, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D topologySize (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);

#define SCHEME_TYPE SchemeType::Dim3
#define ANGLES 0, 0, 0
#endif /* GRID_3D */

  ParallelGridCore parallelGridCore (rank, numProcs, overallSize, false, topologySize);
  ParallelGrid::initializeParallelCore (&parallelGridCore);

  bool isDoubleMaterialPrecision = false;

  ParallelYeeGridLayout<SCHEME_TYPE, LayoutType::E_CENTERED> yeeLayout (overallSize, pmlSize, tfsfSize, ANGLES, isDoubleMaterialPrecision);
  yeeLayout.Initialize (&parallelGridCore);

  ParallelGrid *grid = initGrid (overallSize, bufferSize, yeeLayout.getSizeForCurNode (), false);

  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    bool doSend = ParallelGrid::getParallelCore ()->getDoShare ()[buf].first;
    bool doReceive = ParallelGrid::getParallelCore ()->getDoShare ()[buf].second;

    BufferPosition opposite = ParallelGrid::getParallelCore ()->getOppositeDirections ()[buf];

    int processTo = ParallelGrid::getParallelCore ()->getNodeForDirection ((BufferPosition) buf);
    int processFrom = ParallelGrid::getParallelCore ()->getNodeForDirection (opposite);

    if (doSend && processTo != -1
        && !doReceive)
    {
      ParallelGridCoordinate sendStart = grid->getSendStart ()[buf];
      ParallelGridCoordinate sendEnd = grid->getSendEnd ()[buf];

#ifdef GRID_1D
      grid_coord coordStart[1];
      grid_coord coordEnd[1];
      coordStart[0] = sendStart.get1 ();
      coordEnd[0] = sendEnd.get1 ();
      int size = 1;
#endif
#ifdef GRID_2D
      grid_coord coordStart[2];
      grid_coord coordEnd[2];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordEnd[0] = sendStart.get1 ();
      coordEnd[1] = sendStart.get2 ();
      int size = 2;
#endif
#ifdef GRID_3D
      grid_coord coordStart[3];
      grid_coord coordEnd[3];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordStart[2] = sendStart.get3 ();
      int size = 3;
#endif
      MPI_Send (coordStart, size, MPI_COORD, processTo, rank, ParallelGrid::getParallelCore ()->getCommunicator ());
      MPI_Send (coordEnd, size, MPI_COORD, processTo, rank, ParallelGrid::getParallelCore ()->getCommunicator ());
    }
    else if (doSend && processTo != -1
             && doReceive && processFrom != -1)
    {
      ParallelGridCoordinate sendStart = grid->getSendStart ()[buf];
      ParallelGridCoordinate sendEnd = grid->getSendEnd ()[buf];

#ifdef GRID_1D
      grid_coord coordStart[1];
      grid_coord coordEnd[1];
      coordStart[0] = sendStart.get1 ();
      coordEnd[0] = sendEnd.get1 ();
      int size = 1;
#endif
#ifdef GRID_2D
      grid_coord coordStart[2];
      grid_coord coordEnd[2];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordEnd[0] = sendStart.get1 ();
      coordEnd[1] = sendStart.get2 ();
      int size = 2;
#endif
#ifdef GRID_3D
      grid_coord coordStart[3];
      grid_coord coordEnd[3];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordStart[2] = sendStart.get3 ();
      int size = 3;
#endif

      MPI_Status status;

#ifdef GRID_1D
      grid_coord _coordStart[1];
      grid_coord _coordEnd[1];
      int _size = 1;
#endif
#ifdef GRID_2D
      grid_coord _coordStart[2];
      grid_coord _coordEnd[2];
      int _size = 2;
#endif
#ifdef GRID_3D
      grid_coord _coordStart[3];
      grid_coord _coordEnd[3];
      int _size = 3;
#endif

      MPI_Sendrecv (coordStart, size, MPI_COORD, processTo, rank,
                    _coordStart, _size, MPI_COORD, processFrom, processFrom,
                    ParallelGrid::getParallelCore ()->getCommunicator (), &status);
      MPI_Sendrecv (coordEnd, size, MPI_COORD, processTo, rank,
                    _coordEnd, _size, MPI_COORD, processFrom, processFrom,
                    ParallelGrid::getParallelCore ()->getCommunicator (), &status);

      neighbourSendStart[opposite] = grid->getSendStart ()[buf];
      neighbourSendEnd[opposite] = grid->getSendStart ()[buf];

#ifdef GRID_1D
      neighbourSendStart[opposite].set1 (_coordStart[0]);
      neighbourSendEnd[opposite].set1 (_coordEnd[0]);
#endif
#ifdef GRID_2D
      neighbourSendStart[opposite].set1 (_coordStart[0]);
      neighbourSendStart[opposite].set2 (_coordStart[1]);
      neighbourSendEnd[opposite].set1 (_coordEnd[0]);
      neighbourSendEnd[opposite].set2 (_coordEnd[1]);
#endif
#ifdef GRID_3D
      neighbourSendStart[opposite].set1 (_coordStart[0]);
      neighbourSendStart[opposite].set2 (_coordStart[1]);
      neighbourSendStart[opposite].set3 (_coordStart[2]);
      neighbourSendEnd[opposite].set1 (_coordEnd[0]);
      neighbourSendEnd[opposite].set2 (_coordEnd[1]);
      neighbourSendEnd[opposite].set3 (_coordEnd[2]);
#endif
    }
    else if (!doSend
             && doReceive && processFrom != -1)
    {
      MPI_Status status;

#ifdef GRID_1D
      grid_coord coordStart[1];
      grid_coord coordEnd[1];
      int size = 1;
#endif
#ifdef GRID_2D
      grid_coord coordStart[2];
      grid_coord coordEnd[2];
      int size = 2;
#endif
#ifdef GRID_3D
      grid_coord coordStart[3];
      grid_coord coordEnd[3];
      int size = 3;
#endif
      MPI_Recv (coordStart, size, MPI_COORD, processFrom, processFrom, ParallelGrid::getParallelCore ()->getCommunicator (), &status);
      MPI_Recv (coordEnd, size, MPI_COORD, processFrom, processFrom, ParallelGrid::getParallelCore ()->getCommunicator (), &status);

      neighbourSendStart[opposite] = grid->getSendStart ()[buf];
      neighbourSendEnd[opposite] = grid->getSendStart ()[buf];

#ifdef GRID_1D
      neighbourSendStart[opposite].set1 (coordStart[0]);
      neighbourSendEnd[opposite].set1 (coordEnd[0]);
#endif
#ifdef GRID_2D
      neighbourSendStart[opposite].set1 (coordStart[0]);
      neighbourSendStart[opposite].set2 (coordStart[1]);
      neighbourSendEnd[opposite].set1 (coordEnd[0]);
      neighbourSendEnd[opposite].set2 (coordEnd[1]);
#endif
#ifdef GRID_3D
      neighbourSendStart[opposite].set1 (coordStart[0]);
      neighbourSendStart[opposite].set2 (coordStart[1]);
      neighbourSendStart[opposite].set3 (coordStart[2]);
      neighbourSendEnd[opposite].set1 (coordEnd[0]);
      neighbourSendEnd[opposite].set2 (coordEnd[1]);
      neighbourSendEnd[opposite].set3 (coordEnd[2]);
#endif
    }

    MPI_Barrier (MPI_COMM_WORLD);
  }

  grid->share ();

  /*
   * Check that buffers were correctly initialized
   */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
      {
#endif /* GRID_3D */
#ifdef GRID_1D
        GridCoordinate1D pos (i, CoordinateType::X);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j, CoordinateType::X, CoordinateType::Y);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
#endif /* GRID_3D */

        checkVal (grid, pos);

#if defined (GRID_3D)
      }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  }
#endif /* GRID_1D || GRID_2D || GRID_3D */

  delete grid;
  grid = initGrid (overallSize, bufferSize, yeeLayout.getSizeForCurNode (), true);

  ParallelGridBase *gridTotal = grid->gatherFullGrid ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
      {
#endif /* GRID_3D */

#ifdef GRID_1D
        GridCoordinate1D pos (i, CoordinateType::X);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j, CoordinateType::X, CoordinateType::Y);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
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
        grid_coord step = gridTotal->getSize ().get1 () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        int process = i / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
        grid_coord step = gridTotal->getSize ().get2 () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        int process = j / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
        grid_coord step = gridTotal->getSize ().get3 () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();
        int process = k / step;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
        grid_coord stepX = gridTotal->getSize ().get1 () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepY = gridTotal->getSize ().get2 () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();

        int processI = i / stepX;
        int processJ = j / stepY;

        int process = processJ * ParallelGrid::getParallelCore ()->getNodeGridSizeX () + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
        grid_coord stepY = gridTotal->getSize ().get2 () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        grid_coord stepZ = gridTotal->getSize ().get3 () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processJ = j / stepY;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeY () + processJ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
        grid_coord stepX = gridTotal->getSize ().get1 () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepZ = gridTotal->getSize ().get3 () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processI = i / stepX;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeX () + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
        grid_coord stepX = gridTotal->getSize ().get1 () / ParallelGrid::getParallelCore ()->getNodeGridSizeX ();
        grid_coord stepY = gridTotal->getSize ().get2 () / ParallelGrid::getParallelCore ()->getNodeGridSizeY ();
        grid_coord stepZ = gridTotal->getSize ().get3 () / ParallelGrid::getParallelCore ()->getNodeGridSizeZ ();

        int processI = i / stepX;
        int processJ = j / stepY;
        int processK = k / stepZ;

        int process = processK * ParallelGrid::getParallelCore ()->getNodeGridSizeXY ()
                      + processJ * ParallelGrid::getParallelCore ()->getNodeGridSizeX ()
                      + processI;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

        FPValue fpprocess = (FPValue) process;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
        fpprocess *= i;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
        fpprocess *= j;
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        fpprocess *= k;
#endif /* GRID_3D */

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

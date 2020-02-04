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

#ifndef PARALLEL_GRID
#error This unit tests does not support non-parallel grid mode
#endif /* !PARALLEL_GRID */

#include "ParallelGrid.h"
#include "ParallelYeeGridLayout.h"
#include <mpi.h>

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

const FPValue imagMult = 1000;
const FPValue prevMult = 16;
const FPValue prevPrevMult = prevMult * prevMult;

std::vector<ParallelGridCoordinate> neighborSendStart (BUFFER_COUNT);
std::vector<ParallelGridCoordinate> neighborSendEnd (BUFFER_COUNT);

void checkVal (ParallelGrid *grid, ParallelGridCoordinate pos)
{
  BufferPosition dir = grid->getBufferForPosition (pos);

  if (dir == BUFFER_NONE)
  {
    return;
  }

  BufferPosition opposite = ParallelGrid::getParallelCore ()->getOppositeDirections ()[dir];
  ParallelGridCoordinate posInNeighbor = pos - grid->getRecvStart (opposite) + neighborSendStart[dir];

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
  ASSERT (pidSender != PID_NONE);

  ParallelGridCoordinate sendStart = grid->getSendStart (dir);

  FieldValue cur = *grid->getFieldValue (pos, 0);
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (cur.real () == pidSender * multiplier);
  ASSERT (cur.imag () == pidSender * multiplier * imagMult);
#else
  ASSERT (cur == pidSender * multiplier);
#endif

  FieldValue prev = *grid->getFieldValue (pos, 1);
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (prev.real () == pidSender * multiplier * prevMult);
  ASSERT (prev.imag () == pidSender * multiplier * prevMult * imagMult);
#else
  ASSERT (prev == pidSender * multiplier * prevMult);
#endif

  FieldValue prevPrev = *grid->getFieldValue (pos, 2);
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (prevPrev.real () == pidSender * multiplier * prevPrevMult);
  ASSERT (prevPrev.imag () == pidSender * multiplier * prevPrevMult * imagMult);
#else
  ASSERT (prevPrev == pidSender * multiplier * prevPrevMult);
#endif
}

ParallelGrid * initGrid (ParallelGridCoordinate overallSize,
                         ParallelGridCoordinate bufferSize,
                         ParallelGridCoordinate sizeForCurNode,
                         bool isAbsVal)
{
  ParallelGrid *grid = new ParallelGrid (overallSize, bufferSize, 1, sizeForCurNode, 3, 0);

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
#endif /* GRID_1D || GRID_2D || GRID_3D */
  {
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
#endif /* GRID_2D || GRID_3D */
    {
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
#endif /* GRID_3D */
      {
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

        grid->setFieldValue (FIELDVALUE (fpval, fpval * imagMult), pos, 0);
        grid->setFieldValue (FIELDVALUE (fpval * prevMult, fpval * prevMult * imagMult), pos, 1);
        grid->setFieldValue (FIELDVALUE (fpval * prevPrevMult, fpval * prevPrevMult * imagMult), pos, 2);
      }
    }
  }

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
  DPRINTF (LOG_LEVEL_STAGES, "X: PID %d of %d, grid size x: %llu\n", rank, numProcs, (unsigned long long)gridSizeX);
  ASSERT (gridSizeX % numProcs == 0);
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
  DPRINTF (LOG_LEVEL_STAGES, "Y: PID %d of %d, grid size y: %llu\n", rank, numProcs, (unsigned long long)gridSizeY);
  ASSERT (gridSizeY % numProcs == 0);
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
  DPRINTF (LOG_LEVEL_STAGES, "Z: PID %d of %d, grid size x: %llu\n", rank, numProcs, (unsigned long long)gridSizeZ);
  ASSERT (gridSizeZ % numProcs == 0);
#endif /* GRID_3D */

  DPRINTF (LOG_LEVEL_STAGES, "Start process %d of %d\n", rank, numProcs);

#ifdef GRID_1D
  GridCoordinate1D overallSize (gridSizeX, CoordinateType::X);
  GridCoordinate1D pmlSize (10, CoordinateType::X);
  GridCoordinate1D tfsfSizeLeft (20, CoordinateType::X);
  GridCoordinate1D tfsfSizeRight (20, CoordinateType::X);
  GridCoordinate1D bufferSize (bufSize, CoordinateType::X);
  GridCoordinate1D topologySize (0, CoordinateType::X);

#define SCHEME_TYPE (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy))
#define ANGLES PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2
#endif /* GRID_1D */

#ifdef GRID_2D
  GridCoordinate2D overallSize (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D pmlSize (10, 10, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D tfsfSizeLeft (20, 20, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D tfsfSizeRight (20, 20, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D bufferSize (bufSize, bufSize, CoordinateType::X, CoordinateType::Y);
  GridCoordinate2D topologySize (0, 0, CoordinateType::X, CoordinateType::Y);

#define SCHEME_TYPE (static_cast<SchemeType_t> (SchemeType::Dim2_TEz))
#define ANGLES PhysicsConst::Pi / 2, 0, 0
#endif /* GRID_2D */

#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D pmlSize (10, 10, 10, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D tfsfSizeLeft (20, 20, 20, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D tfsfSizeRight (20, 20, 20, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D bufferSize (bufSize, bufSize, bufSize, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  GridCoordinate3D topologySize (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);

#define SCHEME_TYPE (static_cast<SchemeType_t> (SchemeType::Dim3))
#define ANGLES 0, 0, 0
#endif /* GRID_3D */

  ParallelGridCore parallelGridCore (rank, numProcs, overallSize, false, topologySize);
  ParallelGrid::initializeParallelCore (&parallelGridCore);

  bool isDoubleMaterialPrecision = false;

  ParallelYeeGridLayout<SCHEME_TYPE, E_CENTERED> yeeLayout (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ANGLES, isDoubleMaterialPrecision);
  yeeLayout.Initialize (&parallelGridCore);

  ParallelGrid *grid = initGrid (overallSize, bufferSize, yeeLayout.getSizeForCurNode (), false);

  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    BufferPosition opposite = ParallelGrid::getParallelCore ()->getOppositeDirections ()[buf];

    int processTo = ParallelGrid::getParallelCore ()->getNodeForDirection ((BufferPosition) buf);
    int processFrom = ParallelGrid::getParallelCore ()->getNodeForDirection (opposite);

    if (processTo != PID_NONE
        && processFrom == PID_NONE)
    {
      ParallelGridCoordinate sendStart = grid->getSendStart (buf);
      ParallelGridCoordinate sendEnd = grid->getSendEnd (buf);

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
      coordEnd[0] = sendEnd.get1 ();
      coordEnd[1] = sendEnd.get2 ();
      int size = 2;
#endif
#ifdef GRID_3D
      grid_coord coordStart[3];
      grid_coord coordEnd[3];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordStart[2] = sendStart.get3 ();
      coordEnd[0] = sendEnd.get1 ();
      coordEnd[1] = sendEnd.get2 ();
      coordEnd[2] = sendEnd.get3 ();
      int size = 3;
#endif
      MPI_Send (coordStart, size, MPI_COORD, processTo, rank, ParallelGrid::getParallelCore ()->getCommunicator ());
      MPI_Send (coordEnd, size, MPI_COORD, processTo, rank, ParallelGrid::getParallelCore ()->getCommunicator ());
    }
    else if (processTo != PID_NONE
             && processFrom != PID_NONE)
    {
      ParallelGridCoordinate sendStart = grid->getSendStart (buf);
      ParallelGridCoordinate sendEnd = grid->getSendEnd (buf);

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
      coordEnd[0] = sendEnd.get1 ();
      coordEnd[1] = sendEnd.get2 ();
      int size = 2;
#endif
#ifdef GRID_3D
      grid_coord coordStart[3];
      grid_coord coordEnd[3];
      coordStart[0] = sendStart.get1 ();
      coordStart[1] = sendStart.get2 ();
      coordStart[2] = sendStart.get3 ();
      coordEnd[0] = sendEnd.get1 ();
      coordEnd[1] = sendEnd.get2 ();
      coordEnd[2] = sendEnd.get3 ();
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

      neighborSendStart[opposite] = grid->getSendStart (buf);
      neighborSendEnd[opposite] = grid->getSendStart (buf);

#ifdef GRID_1D
      neighborSendStart[opposite].set1 (_coordStart[0]);
      neighborSendEnd[opposite].set1 (_coordEnd[0]);
#endif
#ifdef GRID_2D
      neighborSendStart[opposite].set1 (_coordStart[0]);
      neighborSendStart[opposite].set2 (_coordStart[1]);
      neighborSendEnd[opposite].set1 (_coordEnd[0]);
      neighborSendEnd[opposite].set2 (_coordEnd[1]);
#endif
#ifdef GRID_3D
      neighborSendStart[opposite].set1 (_coordStart[0]);
      neighborSendStart[opposite].set2 (_coordStart[1]);
      neighborSendStart[opposite].set3 (_coordStart[2]);
      neighborSendEnd[opposite].set1 (_coordEnd[0]);
      neighborSendEnd[opposite].set2 (_coordEnd[1]);
      neighborSendEnd[opposite].set3 (_coordEnd[2]);
#endif
    }
    else if (processTo == PID_NONE
             && processFrom != PID_NONE)
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

      neighborSendStart[opposite] = grid->getSendStart (buf);
      neighborSendEnd[opposite] = grid->getSendStart (buf);

#ifdef GRID_1D
      neighborSendStart[opposite].set1 (coordStart[0]);
      neighborSendEnd[opposite].set1 (coordEnd[0]);
#endif
#ifdef GRID_2D
      neighborSendStart[opposite].set1 (coordStart[0]);
      neighborSendStart[opposite].set2 (coordStart[1]);
      neighborSendEnd[opposite].set1 (coordEnd[0]);
      neighborSendEnd[opposite].set2 (coordEnd[1]);
#endif
#ifdef GRID_3D
      neighborSendStart[opposite].set1 (coordStart[0]);
      neighborSendStart[opposite].set2 (coordStart[1]);
      neighborSendStart[opposite].set3 (coordStart[2]);
      neighborSendEnd[opposite].set1 (coordEnd[0]);
      neighborSendEnd[opposite].set2 (coordEnd[1]);
      neighborSendEnd[opposite].set3 (coordEnd[2]);
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
#endif /* GRID_1D || GRID_2D || GRID_3D */
  {
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
#endif /* GRID_2D || GRID_3D */
    {
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
#endif /* GRID_3D */
      {
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
      }
    }
  }

  delete grid;
  grid = initGrid (overallSize, bufferSize, yeeLayout.getSizeForCurNode (), true);

  ParallelGridBase *gridTotal = grid->gatherFullGrid ();

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
#endif /* GRID_1D || GRID_2D || GRID_3D */
  {
#if defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
#endif /* GRID_2D || GRID_3D */
    {
#if defined (GRID_3D)
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
#endif /* GRID_3D */
      {

#ifdef GRID_1D
        GridCoordinate1D pos (i, CoordinateType::X);
#endif /* GRID_1D */

#ifdef GRID_2D
        GridCoordinate2D pos (i, j, CoordinateType::X, CoordinateType::Y);
#endif /* GRID_2D */

#ifdef GRID_3D
        GridCoordinate3D pos (i, j, k, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
#endif /* GRID_3D */

        FPValue fpval;

#ifdef COMPLEX_FIELD_VALUES

        fpval = gridTotal->getFieldValue (pos, 0)->real ();
        ASSERT (fpval * imagMult == gridTotal->getFieldValue (pos, 0)->imag ());
        ASSERT (fpval * prevMult == gridTotal->getFieldValue (pos, 1)->real ());
        ASSERT (fpval * prevMult * imagMult == gridTotal->getFieldValue (pos, 1)->imag ());
        ASSERT (fpval * prevPrevMult == gridTotal->getFieldValue (pos, 2)->real ());
        ASSERT (fpval * prevPrevMult * imagMult == gridTotal->getFieldValue (pos, 2)->imag ());

#else /* COMPLEX_FIELD_VALUES */

        fpval = *gridTotal->getFieldValue (pos, 0);
        ASSERT (fpval * prevMult == *gridTotal->getFieldValue (pos, 1));
        ASSERT (fpval * prevPrevMult == *gridTotal->getFieldValue (pos, 2));

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
      }
    }
  }

  delete gridTotal;

  MPI_Finalize();

  return 0;
} /* main */

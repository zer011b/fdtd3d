// #include <iostream>
//
// #include "Grid.h"
// #include "Assert.h"
//
// extern const char* BufferPositionNames[];
//
// #if defined (PARALLEL_GRID)
// Grid::Grid (const GridCoordinate& totSize,
//             const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
//             const int process, const int totalProc, uint32_t step) :
//   totalSize (totSize),
//   bufferSizeLeft (bufSizeL),
//   bufferSizeRight (bufSizeR),
//   processId (process),
//   totalProcCount (totalProc),
//   timeStep (step)
// {
//   ASSERT (bufferSizeLeft == bufferSizeRight);
//
// #if defined (ONE_TIME_STEP)
//   grid_iter numTimeStepsInBuild = 2;
// #endif
// #if defined (TWO_TIME_STEPS)
//   grid_iter numTimeStepsInBuild = 3;
// #endif
//
//   oppositeDirections.resize (BUFFER_COUNT);
//   for (int i = 0; i < BUFFER_COUNT; ++i)
//   {
//     oppositeDirections[i] = getOpposite ((BufferPosition) i);
//   }
//
//   sendStart.resize (BUFFER_COUNT);
//   sendEnd.resize (BUFFER_COUNT);
//   recvStart.resize (BUFFER_COUNT);
//   recvEnd.resize (BUFFER_COUNT);
//
//   directions.resize (BUFFER_COUNT);
//
//   buffersSend.resize (BUFFER_COUNT);
//   buffersReceive.resize (BUFFER_COUNT);
//
//   // Call specific constructor.
//   ParallelGridConstructor (numTimeStepsInBuild);
//
//   doShare.resize (BUFFER_COUNT);
//   for (int i = 0; i < BUFFER_COUNT; ++i)
//   {
//     getShare ((BufferPosition) i, doShare[i]);
//   }
//
//   SendReceiveCoordinatesInit ();
//
//   gridValues.resize (size.calculateTotalCoord ());
//
// #if PRINT_MESSAGE
//   printf ("New grid for proc: %d (of %d) with raw size: %lu.\n", process, totalProcCount, gridValues.size ());
// #endif
// }
// #else /* PARALLEL_GRID */
//
// #endif /* !PARALLEL_GRID */
//
// Grid::~Grid ()
// {
//   for (FieldPointValue* current : gridValues)
//   {
//     if (current)
//     {
//       delete current;
//     }
//   }
// }
//
// const GridCoordinate& Grid::getSize () const
// {
//   return size;
// }
//
// VectorFieldPointValues& Grid::getValues ()
// {
//   return gridValues;
// }
//
// bool
// Grid::isLegitIndex (const GridCoordinate& position, const GridCoordinate& sizeCoord) const
// {
// #if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& px = position.getX ();
//   const grid_coord& sx = sizeCoord.getX ();
// #if defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& py = position.getY ();
//   const grid_coord& sy = sizeCoord.getY ();
// #if defined (GRID_3D)
//   const grid_coord& pz = position.getZ ();
//   const grid_coord& sz = sizeCoord.getZ ();
// #endif /* GRID_3D */
// #endif /* GRID_2D || GRID_3D */
// #endif /* GRID_1D || GRID_2D || GRID_3D*/
//
// #if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
//   if (px < 0 || px >= sx)
//   {
//     return false;
//   }
// #if defined (GRID_2D) || defined (GRID_3D)
//   else if (py < 0 || py >= sy)
//   {
//     return false;
//   }
// #if defined (GRID_3D)
//   else if (pz < 0 || pz >= sz)
//   {
//     return false;
//   }
// #endif /* GRID_3D */
// #endif /* GRID_2D || GRID_3D */
// #endif /* GRID_1D || GRID_2D || GRID_3D*/
//
//   return true;
// }
//
// bool
// Grid::isLegitIndex (const GridCoordinate& position) const
// {
//   return isLegitIndex (position, size);
// }
//
// grid_iter
// Grid::calculateIndexFromPosition (const GridCoordinate& position,
//                                           const GridCoordinate& sizeCoord) const
// {
// #if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& px = position.getX ();
//   const grid_coord& sx = sizeCoord.getX ();
// #if defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& py = position.getY ();
//   const grid_coord& sy = sizeCoord.getY ();
// #if defined (GRID_3D)
//   const grid_coord& pz = position.getZ ();
//   const grid_coord& sz = sizeCoord.getZ ();
// #endif /* GRID_3D */
// #endif /* GRID_2D || GRID_3D */
// #endif /* GRID_1D || GRID_2D || GRID_3D*/
//
//   grid_coord coord = 0;
//
// #if defined (GRID_1D)
//   coord = px;
// #else /* GRID_1D */
// #if defined (GRID_2D)
//   coord = px * sy + py;
// #else /* GRID_2D */
// #if defined (GRID_3D)
//   coord = px * sy * sz + py * sz + pz;
// #endif /* GRID_3D */
// #endif /* !GRID_2D */
// #endif /* !GRID_1D */
//
//   return coord;
// }
//
// grid_iter
// Grid::calculateIndexFromPosition (const GridCoordinate& position) const
// {
//   return calculateIndexFromPosition (position, size);
// }
//
// GridCoordinate
// Grid::calculatePositionFromIndex (grid_iter index) const
// {
// #if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& sx = size.getX ();
// #if defined (GRID_2D) || defined (GRID_3D)
//   const grid_coord& sy = size.getY ();
// #if defined (GRID_3D)
//   const grid_coord& sz = size.getZ ();
// #endif /* GRID_3D */
// #endif /* GRID_2D || GRID_3D */
// #endif /* GRID_1D || GRID_2D || GRID_3D */
//
// #if defined (GRID_1D)
//   grid_coord x = index;
//   return GridCoordinate (x);
// #else /* GRID_1D */
// #if defined (GRID_2D)
//   grid_coord x = index / sy;
//   index %= sy;
//   grid_coord y = index;
//   return GridCoordinate (x, y);
// #else /* GRID_2D */
// #if defined (GRID_3D)
//   grid_coord tmp = sy * sz;
//   grid_coord x = index / tmp;
//   index %= tmp;
//   grid_coord y = index / sz;
//   index %= sz;
//   grid_coord z = index;
//   return GridCoordinate (x, y, z);
// #endif /* GRID_3D */
// #endif /* !GRID_2D */
// #endif /* !GRID_1D */
// }
//
// #if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
// void
// Grid::setFieldPointValue (FieldPointValue* value, const GridCoordinate& position)
// {
//   ASSERT (isLegitIndex (position));
//   ASSERT (value);
//
//   grid_iter coord = calculateIndexFromPosition (position);
//
//   if (gridValues[coord])
//   {
//     delete gridValues[coord];
//   }
//
//   gridValues[coord] = value;
// }
//
// void
// Grid::setFieldPointValueCurrent (const FieldValue& value,
//                                  const GridCoordinate& position)
// {
//   ASSERT (isLegitIndex (position));
//
//   grid_iter coord = calculateIndexFromPosition (position);
//
//   gridValues[coord]->setCurValue (value);
// }
//
// FieldPointValue*
// Grid::getFieldPointValue (const GridCoordinate& position)
// {
//   ASSERT (isLegitIndex (position));
//
//   grid_iter coord = calculateIndexFromPosition (position);
//   FieldPointValue* value = gridValues[coord];
//
//   ASSERT (value);
//
//   return value;
// }
//
// FieldPointValue*
// Grid::getFieldPointValue (grid_iter coord)
// {
//   ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());
//
//   FieldPointValue* value = gridValues[coord];
//
//   ASSERT (value);
//
//   return value;
// }
//
// /*#if defined (PARALLEL_GRID)
// FieldPointValue*
// Grid::getFieldPointValueGlobal (const GridCoordinate& position)
// {
//   return NULL;
// }
//
// FieldPointValue*
// Grid::getFieldPointValueGlobal (grid_iter coord)
// {
//   return NULL;
// }
// #endif*/
//
// #endif /* GRID_1D || GRID_2D || GRID_3D */
//
// void
// Grid::shiftInTime ()
// {
//   for (FieldPointValue* current : getValues ())
//   {
//     current->shiftInTime ();
//   }
// }
//
// void
// Grid::nextTimeStep ()
// {
//   shiftInTime ();
//
// #if defined (PARALLEL_GRID)
//   grid_coord shiftCounter = bufferSizeLeft.getMax ();
//   if (timeStep % shiftCounter == 0)
//   {
//     Share ();
//   }
// #endif
// }

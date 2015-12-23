#include <iostream>

#include "FieldGrid.h"
#include "Assert.h"

// ================================ GridSize ================================
GridCoordinate::GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx
#if defined (GRID_2D) || defined (GRID_3D)
  , const grid_coord& sy
#if defined (GRID_3D)
  , const grid_coord& sz
#endif
#endif
#endif
  ) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (sx)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (sy)
#if defined (GRID_3D)
  , z (sz)
#endif
#endif
#endif
{
}

GridCoordinate::~GridCoordinate ()
{
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getX () const
{
  return x;
}
#if defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getY () const
{
  return y;
}
#if defined (GRID_3D)
const grid_coord&
GridCoordinate::getZ () const
{
  return z;
}
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_iter
GridCoordinate::calculateTotalCoord () const
{
#if defined (GRID_1D)
  return x;
#else
#if defined (GRID_2D)
  return x * y;
#else
#if defined (GRID_3D)
  return x * y * z;
#endif
#endif
#endif
}
#endif

// ================================ Grid ================================
#if defined (PARALLEL_GRID)
Grid::Grid (const GridCoordinate& totSize, const GridCoordinate& curSize,
            const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
            const int process, const int totalProc) :
  size (curSize + bufSizeL + bufSizeR),
  currentSize (curSize),
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
  totalSize (totSize),
  processId (process),
  totalProcCount (totalProc)
{
  gridValues.resize (size.calculateTotalCoord ());

  //std::cout << "New grid for proc: " << process << " (of " << totalProcCount << ") with size: "
  //  << gridValues.size () << ". " << std::endl;
  printf ("New grid for proc: %d (of %d) with raw size: %lu.\n", process, totalProcCount, gridValues.size ());
}
#else
Grid::Grid(const GridCoordinate& s) :
  size (s)
{
  gridValues.resize (size.calculateTotalCoord ());

  //std::cout << "New grid with raw size: " << gridValues.size () << ". " << std::endl;
  printf ("New grid with raw size: %lu.\n", gridValues.size ());
}
#endif

Grid::~Grid ()
{
  for (FieldPointValue* current : gridValues)
  {
    if (current)
    {
      delete current;
    }
  }
}

const GridCoordinate& Grid::getSize () const
{
  return size;
}

VectorFieldPointValues& Grid::getValues ()
{
  return gridValues;
}

bool
Grid::isLegitIndex (const GridCoordinate& position) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& px = position.getX ();
  const grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& py = position.getY ();
  const grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = size.getZ ();
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  if (px < 0 || px >= sx)
  {
    return false;
  }
#if defined (GRID_2D) || defined (GRID_3D)
  else if (py < 0 || py >= sy)
  {
    return false;
  }
#if defined (GRID_3D)
  else if (pz < 0 || pz >= sz)
  {
    return false;
  }
#endif
#endif
#endif

  return true;
}

grid_iter
Grid::calculateIndexFromPosition (const GridCoordinate& position) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& px = position.getX ();
  const grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& py = position.getY ();
  const grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = size.getZ ();
#endif
#endif
#endif

  grid_coord coord = 0;

#if defined (GRID_1D)
  coord = px;
#else
#if defined (GRID_2D)
  coord = px * sy + py;
#else
#if defined (GRID_3D)
  coord = px * sy * sz + py * sz + pz;
#endif
#endif
#endif

  return coord;
}

GridCoordinate
Grid::calculatePositionFromIndex (grid_iter index) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  const grid_coord& sz = size.getZ ();
#endif
#endif
#endif

#if defined (GRID_1D)
  grid_coord x = index;
  return GridCoordinate (x);
#else
#if defined (GRID_2D)
  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;
  return GridCoordinate (x, y);
#else
#if defined (GRID_3D)
  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;
  return GridCoordinate (x, y, z);
#endif
#endif
#endif
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
void
Grid::setFieldPointValue (FieldPointValue* value, const GridCoordinate& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  if (gridValues[coord])
  {
    delete gridValues[coord];
  }

  gridValues[coord] = value;
}

FieldPointValue*
Grid::getFieldPointValue (const GridCoordinate& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

FieldPointValue*
Grid::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

/*#if defined (PARALLEL_GRID)
FieldPointValue*
Grid::getFieldPointValueGlobal (const GridCoordinate& position)
{
  return NULL;
}

FieldPointValue*
Grid::getFieldPointValueGlobal (grid_iter coord)
{
  return NULL;
}
#endif*/

#endif

#if defined (PARALLEL_GRID)
void
Grid::SendBuffer (BufferPosition buffer, int processTo)
{
#if defined (GRID_1D)
  switch (buffer)
  {
    case LEFT:
    {
      int numTimeValues = sizeof (FieldPointValue) / sizeof (FieldValue);
      FieldValue* raw_values = new FieldValue[bufferSizeLeft.getX () * numTimeValues];

      int j = 0;
      for (int i = bufferSizeLeft.getX (); i < 2*bufferSizeLeft.getX (); ++i)
      {
        FieldPointValue* val = getFieldPointValue (i);
        raw_values[j++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        raw_values[j++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
        raw_values[j++] = val->getPrevPrevValue ();
#endif
#endif
      }

#if FULL_VALUES
      MPI_Ssend (raw_values, bufferSizeLeft.getX () * numTimeValues, MPI_DOUBLE, processTo, processId, MPI_COMM_WORLD);
#else
      MPI_Ssend (raw_values, bufferSizeLeft.getX () * numTimeValues, MPI_FLOAT, processTo, processId, MPI_COMM_WORLD);
#endif

      delete[] raw_values;
      break;
    }
    case RIGHT:
    {
      int numTimeValues = sizeof (FieldPointValue) / sizeof (FieldValue);
      FieldValue* raw_values = new FieldValue[bufferSizeRight.getX () * numTimeValues];

      int j = 0;
      for (int i = size.getX() - 2*bufferSizeRight.getX (); i < size.getX () - bufferSizeRight.getX (); ++i)
      {
        FieldPointValue* val = getFieldPointValue (i);
        raw_values[j++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        raw_values[j++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
        raw_values[j++] = val->getPrevPrevValue ();
#endif
#endif
      }

#if FULL_VALUES
      MPI_Ssend (raw_values, bufferSizeRight.getX () * numTimeValues, MPI_DOUBLE, processTo, processId, MPI_COMM_WORLD);
#else
      MPI_Ssend (raw_values, bufferSizeRight.getX () * numTimeValues, MPI_FLOAT, processTo, processId, MPI_COMM_WORLD);
#endif

      delete[] raw_values;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
#endif
#if defined (GRID_2D)

#endif
#if defined (GRID_3D)

#endif
}

void
Grid::ReceiveBuffer (BufferPosition buffer, int processFrom)
{
#if defined (GRID_1D)
  switch (buffer)
  {
    case LEFT:
    {
      int numTimeValues = sizeof (FieldPointValue) / sizeof (FieldValue);
      FieldValue* raw_values = new FieldValue[bufferSizeLeft.getX () * numTimeValues];

      MPI_Status status;

#if FULL_VALUES
      MPI_Recv (raw_values, bufferSizeLeft.getX () * numTimeValues, MPI_DOUBLE, processFrom, processId, MPI_COMM_WORLD, &status);
#else
      MPI_Recv (raw_values, bufferSizeLeft.getX () * numTimeValues, MPI_FLOAT, processFrom, processId, MPI_COMM_WORLD, &status);
#endif

      int j = 0;
      for (int i = 0; i < bufferSizeLeft.getX (); ++i)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (raw_values[j++], raw_values[j++], raw_values[j++]);
#else
#if defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (raw_values[j++], raw_values[j++]);
#else
        FieldPointValue* val = new FieldPointValue (raw_values[j++]);
#endif
#endif

        setFieldPointValue (val, GridCoordinate (i));
      }

      delete[] raw_values;
      break;
    }
    case RIGHT:
    {
      int numTimeValues = sizeof (FieldPointValue) / sizeof (FieldValue);
      FieldValue* raw_values = new FieldValue[bufferSizeRight.getX () * numTimeValues];

      MPI_Status status;

#if FULL_VALUES
      MPI_Recv (raw_values, bufferSizeRight.getX () * numTimeValues, MPI_DOUBLE, processFrom, processId, MPI_COMM_WORLD, &status);
#else
      MPI_Recv (raw_values, bufferSizeRight.getX () * numTimeValues, MPI_FLOAT, processFrom, processId, MPI_COMM_WORLD, &status);
#endif

      int j = 0;
      for (int i = size.getX() - bufferSizeRight.getX (); i < size.getX(); ++i)
      {
#if defined (TWO_TIME_STEPS)
        FieldPointValue* val = new FieldPointValue (raw_values[j++], raw_values[j++], raw_values[j++]);
#else
#if defined (ONE_TIME_STEP)
        FieldPointValue* val = new FieldPointValue (raw_values[j++], raw_values[j++]);
#else
        FieldPointValue* val = new FieldPointValue (raw_values[j++]);
#endif
#endif

        setFieldPointValue (val, GridCoordinate (i));
      }

      delete[] raw_values;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
#endif
#if defined (GRID_2D)

#endif
#if defined (GRID_3D)

#endif
}
#endif

void
Grid::shiftInTime ()
{
  for (FieldPointValue* current : getValues ())
  {
    current->shiftInTime ();
  }
}

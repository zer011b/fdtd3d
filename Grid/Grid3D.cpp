#include "Grid3D.h"
#include "Assert.h"

#include <cstdlib>

extern const char* BufferPositionNames[];

/**
 * ======== Consructors and destructor ========
 */

Grid3D::Grid3D (const GridCoordinate3D& s, uint32_t step) :
  size (s),
  gridValues (size.calculateTotalCoord ()),
  timeStep (step)
{
  for (int i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = nullptr;
  }

#if PRINT_MESSAGE
  printf ("New grid with raw size: %lu.\n", gridValues.size ());
#endif
}

Grid3D::~Grid3D ()
{
  for (FieldPointValue* i_p : gridValues)
  {
    delete i_p;
  }
}

/**
 * ======== Static methods ========
 */

bool
Grid3D::isLegitIndex (const GridCoordinate3D& position,
                      const GridCoordinate3D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

  if (px < 0 || px >= sx)
  {
    return false;
  }
  else if (py < 0 || py >= sy)
  {
    return false;
  }
  else if (pz < 0 || pz >= sz)
  {
    return false;
  }

  return true;
}

grid_iter
Grid3D::calculateIndexFromPosition (const GridCoordinate3D& position,
                                    const GridCoordinate3D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

  return px * sy * sz + py * sz + pz;
}

/**
 * ======== Private methods ========
 */

VectorFieldPointValues&
Grid3D::getValues ()
{
  return gridValues;
}

void
Grid3D::shiftInTime ()
{
  for (FieldPointValue* i_p : getValues ())
  {
    i_p->shiftInTime ();
  }
}

bool
Grid3D::isLegitIndex (const GridCoordinate3D& position) const
{
  return isLegitIndex (position, size);
}

grid_iter
Grid3D::calculateIndexFromPosition (const GridCoordinate3D& position) const
{
  return calculateIndexFromPosition (position, size);
}

/**
 * ======== Public methods ========
 */

const GridCoordinate3D&
Grid3D::getSize () const
{
  return size;
}

GridCoordinate3D
Grid3D::calculatePositionFromIndex (grid_iter index) const
{
  const grid_coord& sy = size.getY ();
  const grid_coord& sz = size.getZ ();

  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;
  return GridCoordinate3D (x, y, z);
}

void
Grid3D::setFieldPointValue (FieldPointValue* value,
                            const GridCoordinate3D& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  delete gridValues[coord];

  gridValues[coord] = value;
}

void
Grid3D::setFieldPointValueCurrent (const FieldValue& value,
                                   const GridCoordinate3D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);

  gridValues[coord]->setCurValue (value);
}

FieldPointValue*
Grid3D::getFieldPointValue (const GridCoordinate3D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

FieldPointValue*
Grid3D::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

void
Grid3D::nextTimeStep ()
{
  shiftInTime ();
}

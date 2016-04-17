#include "Grid2D.h"
#include "Assert.h"

#include <cstdlib>

extern const char* BufferPositionNames[];

/**
 * ======== Consructors and destructor ========
 */

Grid2D::Grid2D (const GridCoordinate2D& s, uint32_t step) :
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

Grid2D::~Grid2D ()
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
Grid2D::isLegitIndex (const GridCoordinate2D& position,
                      const GridCoordinate2D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  if (px < 0 || px >= sx)
  {
    return false;
  }
  else if (py < 0 || py >= sy)
  {
    return false;
  }

  return true;
}

grid_iter
Grid2D::calculateIndexFromPosition (const GridCoordinate2D& position,
                                    const GridCoordinate2D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  return px * sy + py;
}

/**
 * ======== Private methods ========
 */

VectorFieldPointValues&
Grid2D::getValues ()
{
  return gridValues;
}

void
Grid2D::shiftInTime ()
{
  for (FieldPointValue* i_p : getValues ())
  {
    i_p->shiftInTime ();
  }
}

bool
Grid2D::isLegitIndex (const GridCoordinate2D& position) const
{
  return isLegitIndex (position, size);
}

grid_iter
Grid2D::calculateIndexFromPosition (const GridCoordinate2D& position) const
{
  return calculateIndexFromPosition (position, size);
}

/**
 * ======== Public methods ========
 */

const GridCoordinate2D&
Grid2D::getSize () const
{
  return size;
}

GridCoordinate2D
Grid2D::calculatePositionFromIndex (grid_iter index) const
{
  const grid_coord& sx = size.getX ();
  const grid_coord& sy = size.getY ();

  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;
  return GridCoordinate2D (x, y);
}

void
Grid2D::setFieldPointValue (FieldPointValue* value,
                            const GridCoordinate2D& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  delete gridValues[coord];

  gridValues[coord] = value;
}

void
Grid2D::setFieldPointValueCurrent (const FieldValue& value,
                                   const GridCoordinate2D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);

  gridValues[coord]->setCurValue (value);
}

FieldPointValue*
Grid2D::getFieldPointValue (const GridCoordinate2D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

FieldPointValue*
Grid2D::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

void
Grid2D::nextTimeStep ()
{
  shiftInTime ();
}

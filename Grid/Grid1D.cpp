#include "Grid1D.h"
#include "Assert.h"

#include <cstdlib>

extern const char* BufferPositionNames[];

/**
 * ======== Consructors and destructor ========
 */

Grid1D::Grid1D (const GridCoordinate1D& s, uint32_t step) :
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

Grid1D::~Grid1D ()
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
Grid1D::isLegitIndex (const GridCoordinate1D& position,
                      const GridCoordinate1D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  if (px < 0 || px >= sx)
  {
    return false;
  }

  return true;
}

grid_iter
Grid1D::calculateIndexFromPosition (const GridCoordinate1D& position,
                                    const GridCoordinate1D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  return px;
}

/**
 * ======== Private methods ========
 */

VectorFieldPointValues&
Grid1D::getValues ()
{
  return gridValues;
}

void
Grid1D::shiftInTime ()
{
  for (FieldPointValue* i_p : getValues ())
  {
    i_p->shiftInTime ();
  }
}

bool
Grid1D::isLegitIndex (const GridCoordinate1D& position) const
{
  return isLegitIndex (position, size);
}

grid_iter
Grid1D::calculateIndexFromPosition (const GridCoordinate1D& position) const
{
  return calculateIndexFromPosition (position, size);
}

/**
 * ======== Public methods ========
 */

const GridCoordinate1D&
Grid1D::getSize () const
{
  return size;
}

GridCoordinate1D
Grid1D::calculatePositionFromIndex (grid_iter index) const
{
  return GridCoordinate1D (index);
}

void
Grid1D::setFieldPointValue (FieldPointValue* value,
                            const GridCoordinate1D& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  delete gridValues[coord];

  gridValues[coord] = value;
}

void
Grid1D::setFieldPointValueCurrent (const FieldValue& value,
                                   const GridCoordinate1D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);

  gridValues[coord]->setCurValue (value);
}

FieldPointValue*
Grid1D::getFieldPointValue (const GridCoordinate1D& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

FieldPointValue*
Grid1D::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

void
Grid1D::nextTimeStep ()
{
  shiftInTime ();
}

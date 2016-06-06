#ifndef GRID_H
#define GRID_H

#include <vector>
#include <cstdlib>

#include "Assert.h"
#include "FieldPoint.h"
#include "GridCoordinate3D.h"

/**
 * Vector of points in grid.
 */
typedef std::vector<FieldPointValue*> VectorFieldPointValues;

/**
 * Non-parallel grid class.
 */
template <class TCoord>
class Grid
{
protected:

  // Size of the grid.
  // For parallel grid - size of current node plus size of buffers.
  TCoord size;

  // Vector of points in grid.
  // Owns this. Deletes all FieldPointValue* itself.
  VectorFieldPointValues gridValues;

  // Current time step.
  uint32_t timeStep;

private:

  // Check whether position is appropriate to get/set value from.
  static bool isLegitIndex (const TCoord& position,
                            const TCoord& sizeCoord);

  // Calculate N-dimensional coordinate from position.
  static grid_iter calculateIndexFromPosition (const TCoord& position,
                                               const TCoord& sizeCoord);

private:

  // Get values in the grid.
  VectorFieldPointValues& getValues ();

  // Replace previous layer with current and so on.
  void shiftInTime ();

  // Check whether position is appropriate to get/set value from.
  bool isLegitIndex (const TCoord& position) const;

  // Calculate N-dimensional coordinate from position.
  grid_iter calculateIndexFromPosition (const TCoord& position) const;

public:

  Grid (const TCoord& s, uint32_t step);
  Grid (uint32_t step);
  ~Grid ();

  // Get size of the grid.
  const TCoord& getSize () const;

  // Calculate position from three-dimensional coordinate.
  TCoord calculatePositionFromIndex (grid_iter index) const;

  // Set field point at coordinate in grid.
  void setFieldPointValue (FieldPointValue* value, const TCoord& position);

  // Get field point at coordinate in grid.
  FieldPointValue* getFieldPointValue (const TCoord& position);
  FieldPointValue* getFieldPointValue (grid_iter coord);

  // Switch to next time step.
  virtual void nextTimeStep ();
};

/**
 * ======== Consructors and destructor ========
 */
template <class TCoord>
Grid<TCoord>::Grid (const TCoord& s, uint32_t step) :
  size (s),
  gridValues (size.calculateTotalCoord ()),
  timeStep (step)
{
  for (int i = 0; i < gridValues.size (); ++i)
  {
#ifdef CXX11_ENABLED
    gridValues[i] = nullptr;
#else
    gridValues[i] = NULL;
#endif
  }

#if PRINT_MESSAGE
  printf ("New grid with raw size: %lu.\n", gridValues.size ());
#endif /* PRINT_MESSAGE */
}

template <class TCoord>
Grid<TCoord>::Grid (uint32_t step) :
  timeStep (step)
{
#if PRINT_MESSAGE
  printf ("New grid without size.\n");
#endif /* PRINT_MESSAGE */
}

template <class TCoord>
Grid<TCoord>::~Grid ()
{
#ifdef CXX11_ENABLED
  for (FieldPointValue* i_p : gridValues)
  {
    delete i_p;
  }
#else
  for (VectorFieldPointValues::iterator iter = gridValues.begin ();
       iter != gridValues.end (); ++iter)
  {
    delete (*iter);
  }
#endif

}

/**
 * ======== Private methods ========
 */

template <class TCoord>
VectorFieldPointValues&
Grid<TCoord>::getValues ()
{
  return gridValues;
}

template <class TCoord>
void
Grid<TCoord>::shiftInTime ()
{
#ifdef CXX11_ENABLED
  for (FieldPointValue* i_p : getValues ())
  {
    i_p->shiftInTime ();
  }
#else
  for (VectorFieldPointValues::iterator iter = gridValues.begin ();
       iter != gridValues.end (); ++iter)
  {
    (*iter)->shiftInTime ();
  }
#endif
}

template <class TCoord>
bool
Grid<TCoord>::isLegitIndex (const TCoord& position) const
{
  return isLegitIndex (position, size);
}

template <class TCoord>
grid_iter
Grid<TCoord>::calculateIndexFromPosition (const TCoord& position) const
{
  return calculateIndexFromPosition (position, size);
}

/**
 * ======== Public methods ========
 */

template <class TCoord>
const TCoord&
Grid<TCoord>::getSize () const
{
  return size;
}

template <class TCoord>
void
Grid<TCoord>::setFieldPointValue (FieldPointValue* value,
                                  const TCoord& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  delete gridValues[coord];

  gridValues[coord] = value;
}

template <class TCoord>
FieldPointValue*
Grid<TCoord>::getFieldPointValue (const TCoord& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

template <class TCoord>
FieldPointValue*
Grid<TCoord>::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

template <class TCoord>
void
Grid<TCoord>::nextTimeStep ()
{
  shiftInTime ();
}

#endif /* GRID_H */

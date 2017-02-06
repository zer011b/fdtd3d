#ifndef GRID_H
#define GRID_H

#include <cstdlib>
#include <vector>

#include "Assert.h"
#include "FieldPoint.h"
#include "GridCoordinate3D.h"

/**
 * Type of vector of points in grid.
 */
typedef std::vector<FieldPointValue *> VectorFieldPointValues;

/**
 * Non-parallel grid class.
 */
template <class TCoord>
class Grid
{
protected:

  /**
   * Size of the grid. For parallel grid - size of current node plus size of buffers.
   */
  TCoord size;

  /**
   * Vector of points in grid. Owns this. Deletes all FieldPointValue* itself.
   */
  VectorFieldPointValues gridValues;

  /**
   * Current time step.
   */
  time_step timeStep;

  /*
   * TODO: add grid name
   */
  std::string gridName;

protected:

  static bool isLegitIndex (const TCoord &, const TCoord &);
  static grid_iter calculateIndexFromPosition (const TCoord &, const TCoord &);

private:

  VectorFieldPointValues& getValues ();
  void shiftInTime ();
  bool isLegitIndex (const TCoord &) const;
  grid_iter calculateIndexFromPosition (const TCoord &) const;

public:

  Grid (const TCoord& s, time_step step, const char * = "unnamed");
  Grid (time_step step, const char * = "unnamed");
  ~Grid ();

  const TCoord &getSize () const;
  TCoord getTotalPosition (TCoord) const;
  TCoord getTotalSize () const;
  TCoord getRelativePosition (TCoord) const;

  virtual TCoord getComputationStart () const;
  virtual TCoord getComputationEnd () const;
  TCoord calculatePositionFromIndex (grid_iter) const;

  void setFieldPointValue (FieldPointValue *, const TCoord &);
  virtual FieldPointValue *getFieldPointValue (const TCoord &);
  virtual FieldPointValue *getFieldPointValue (grid_iter);

  virtual void nextTimeStep ();
}; /* Grid */

/*
 * Templates definition
 */

/**
 * Constructor of grid
 */
template <class TCoord>
Grid<TCoord>::Grid (const TCoord &s, /**< size of grid */
                    time_step step, /**< default time step */
                    const char *name) /**< name of grid */
  : size (s)
  , gridValues (size.calculateTotalCoord ())
  , timeStep (step)
  , gridName (name)
{
  for (int i = 0; i < gridValues.size (); ++i)
  {
#ifdef CXX11_ENABLED
    gridValues[i] = nullptr;
#else /* CXX11_ENABLED */
    gridValues[i] = NULL;
#endif /* !CXX11_ENABLED */
  }

#if PRINT_MESSAGE
  printf ("New grid '%s' with raw size: %lu.\n", gridName, gridValues.size ());
#endif /* PRINT_MESSAGE */
} /* Grid<TCoord>::Grid */

/**
 * Constructor of grid without size
 */
template <class TCoord>
Grid<TCoord>::Grid (time_step step, /**< default time step */
                    const char *name) /**< name of grid */
  : timeStep (step)
  , gridName (name)
{
#if PRINT_MESSAGE
  printf ("New grid '%s' without size.\n", gridName);
#endif /* PRINT_MESSAGE */
} /* Grid<TCoord>::Grid */

/**
 * Destructor of grid. Should free all field point values
 */
template <class TCoord>
Grid<TCoord>::~Grid ()
{
#ifdef CXX11_ENABLED
  for (FieldPointValue* i_p : gridValues)
  {
    delete i_p;
  }
#else /* CXX11_ENABLED */
  for (VectorFieldPointValues::iterator iter = gridValues.begin ();
       iter != gridValues.end ();
       ++iter)
  {
    delete (*iter);
  }
#endif /* !CXX11_ENABLED */
} /* Grid<TCoord>::~Grid */

/**
 * Get values of the grid
 *
 * @return values of the grid
 */
template <class TCoord>
VectorFieldPointValues&
Grid<TCoord>::getValues ()
{
  return gridValues;
} /* Grid<TCoord>::getValues */

/**
 * Replace previous time layer with current and so on
 */
template <class TCoord>
void
Grid<TCoord>::shiftInTime ()
{
#ifdef CXX11_ENABLED
  for (FieldPointValue* i_p : getValues ())
  {
    i_p->shiftInTime ();
  }
#else /* CXX11_ENABLED */
  for (VectorFieldPointValues::iterator iter = gridValues.begin ();
       iter != gridValues.end ();
       ++iter)
  {
    (*iter)->shiftInTime ();
  }
#endif /* !CXX11_ENABLED */
} /* Grid<TCoord>::shiftInTime */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template <class TCoord>
bool
Grid<TCoord>::isLegitIndex (const TCoord& position) const /**< coordinate in grid */
{
  return isLegitIndex (position, size);
} /* Grid<TCoord>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from N-dimensional position
 *
 * @return one-dimensional coordinate from N-dimensional position
 */
template <class TCoord>
grid_iter
Grid<TCoord>::calculateIndexFromPosition (const TCoord& position) const /**< coordinate in grid */
{
  return calculateIndexFromPosition (position, size);
} /* Grid<TCoord>::calculateIndexFromPosition */

/**
 * Get size of the grid
 *
 * @return size of the grid
 */
template <class TCoord>
const TCoord&
Grid<TCoord>::getSize () const
{
  return size;
} /* Grid<TCoord>::getSize */

/**
 * Get first coordinate from which to perfrom computations at current step
 *
 * @return first coordinate from which to perfrom computations at current step
 */
template <class TCoord>
TCoord
Grid<TCoord>::getComputationStart () const
{
  return TCoord (0);
} /* Grid<TCoord>::getComputationStart */

/**
 * Get last coordinate until which to perfrom computations at current step
 *
 * @return last coordinate until which to perfrom computations at current step
 */
template <class TCoord>
TCoord
Grid<TCoord>::getComputationEnd () const
{
  return getSize ();
} /* Grid<TCoord>::getComputationEnd () */

/**
 * Set field point value at coordinate in grid
 */
template <class TCoord>
void
Grid<TCoord>::setFieldPointValue (FieldPointValue *value, /**< field point value */
                                  const TCoord &position) /**< coordinate in grid */
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  delete gridValues[coord];

  gridValues[coord] = value;
} /* Grid<TCoord>::setFieldPointValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
FieldPointValue *
Grid<TCoord>::getFieldPointValue (const TCoord &position) /**< cooridnate in grid */
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);

  return getFieldPointValue (coord);
} /* Grid<TCoord>::getFieldPointValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
FieldPointValue *
Grid<TCoord>::getFieldPointValue (grid_iter coord) /**< index in grid */
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
} /* Grid<TCoord>::getFieldPointValue */

/**
 * Switch to next time step
 */
template <class TCoord>
void
Grid<TCoord>::nextTimeStep ()
{
  shiftInTime ();
} /* Grid<TCoord>::nextTimeStep */

/**
 * Get total position in grid. Is equal to position in non-parallel grid
 *
 * @return total position in grid
 */
template <class TCoord>
TCoord
Grid<TCoord>::getTotalPosition (TCoord pos) const /**< position in grid */
{
  return TCoord (0) + pos;
} /* Grid<TCoord>::getTotalPosition */

/**
 * Get total size of grid. Is equal to size in non-parallel grid
 *
 * @return total size of grid
 */
template <class TCoord>
TCoord
Grid<TCoord>::getTotalSize () const
{
  return getSize ();
} /* Grid<TCoord>::getTotalSize */

/**
 * Get relative position in grid. Is equal to position in non-parallel grid
 *
 * @return relative position in grid
 */
template <class TCoord>
TCoord
Grid<TCoord>::getRelativePosition (TCoord pos) const /**< position in grid */
{
  return pos - TCoord (0);
} /* gGrid<TCoord>::etRelativePosition */

#endif /* GRID_H */

#ifndef GRID_H
#define GRID_H

#include <cstdlib>
#include <vector>
#include <string>

#include "Assert.h"
#include "FieldPoint.h"
#include "GridCoordinate3D.h"
#include "Settings.h"

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

  /**
   * Name of the grid.
   */
  std::string gridName;

  /*
   * TODO: add debug uninitialized flag
   */

protected:

  static bool isLegitIndex (const TCoord &, const TCoord &);
  static grid_coord calculateIndexFromPosition (const TCoord &, const TCoord &);

private:

  VectorFieldPointValues& getValues ();
  void shiftInTime ();

  void deleteGrid ();
  void copyGrid (const Grid &);

protected:

  bool isLegitIndex (const TCoord &) const;
  grid_coord calculateIndexFromPosition (const TCoord &) const;

public:

  Grid (const TCoord& s, time_step step, const char * = "unnamed");
  Grid (time_step step, const char * = "unnamed");
  Grid (const Grid &grid);
  virtual ~Grid ();

  Grid<TCoord> & operator = (const Grid<TCoord> &grid);

  const TCoord &getSize () const;
  virtual TCoord getTotalPosition (TCoord) const;
  virtual TCoord getTotalSize () const;
  virtual TCoord getRelativePosition (TCoord) const;

  virtual TCoord getComputationStart (TCoord) const;
  virtual TCoord getComputationEnd (TCoord) const;
  TCoord calculatePositionFromIndex (grid_coord) const;

  void setFieldPointValue (FieldPointValue *, const TCoord &);
  FieldPointValue *getFieldPointValue (const TCoord &);
  FieldPointValue *getFieldPointValue (grid_coord);

  virtual FieldPointValue *getFieldPointValueByAbsolutePos (const TCoord &);
  virtual FieldPointValue *getFieldPointValueOrNullByAbsolutePos (const TCoord &);

  virtual void nextTimeStep ();

  const std::string &getName () const;

  void initialize ();
  void initialize (FieldValue);
}; /* Grid */

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
    gridValues[i] = NULLPTR;
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' with raw size: %lu.\n", gridName.data (), gridValues.size ());
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
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' without size.\n", gridName.data ());
} /* Grid<TCoord>::Grid */

/**
 * Copy constructor
 */
template <class TCoord>
Grid<TCoord>::Grid (const Grid<TCoord> &grid) /**< grid to copy */
{
  copyGrid (grid);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New copied grid '%s' with raw size: %lu.\n", gridName.data (), gridValues.size ());
} /* Grid<TCoord>::Grid */

/**
 * Destructor of grid. Should delete all field point values
 */
template <class TCoord>
Grid<TCoord>::~Grid ()
{
  deleteGrid ();
} /* Grid<TCoord>::~Grid */

/**
 * Delete grid
 */
template <class TCoord>
void
Grid<TCoord>::deleteGrid ()
{
  for (grid_coord i = 0; i < gridValues.size (); ++i)
  {
    delete gridValues[i];
    gridValues[i] = NULLPTR;
  }
} /* Grid<TCoord>::deleteGrid */

/**
 * Copy one grid to another
 */
template <class TCoord>
void
Grid<TCoord>::copyGrid (const Grid<TCoord> &grid) /**< grid to copy */
{
  size = grid.size;
  gridValues.resize (grid.gridValues.size ());
  timeStep = grid.timeStep;
  gridName = grid.gridName;

  for (grid_coord i = 0; i < grid.gridValues.size (); ++i)
  {
    if (grid.gridValues[i])
    {
      gridValues[i] = new FieldPointValue ();
      *gridValues[i] = *grid.gridValues[i];
    }
    else
    {
      gridValues[i] = NULLPTR;
    }
  }
} /* Grid<TCoord>::copyGrid */

/**
 * Operator =
 */
template <class TCoord>
Grid<TCoord> &
Grid<TCoord>::operator = (const Grid<TCoord> &grid) /**< grid to assign */
{
  deleteGrid ();
  copyGrid (grid);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Copied grid '%s' with raw size: %lu.\n", gridName.data (), gridValues.size ());

  return *this;
} /* Grid<TCoord>::operator= */

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
  for (VectorFieldPointValues::iterator iter = gridValues.begin ();
       iter != gridValues.end ();
       ++iter)
  {
    ASSERT (*iter != NULLPTR);
    (*iter)->shiftInTime ();
  }
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
grid_coord
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
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <class TCoord>
TCoord
Grid<TCoord>::getComputationEnd (TCoord diffPosEnd) const
{
  return getSize () - diffPosEnd;
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

  grid_coord coord = calculateIndexFromPosition (position);

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
Grid<TCoord>::getFieldPointValue (const TCoord &position) /**< coordinate in grid */
{
  ASSERT (isLegitIndex (position));

  grid_coord coord = calculateIndexFromPosition (position);

  return getFieldPointValue (coord);
} /* Grid<TCoord>::getFieldPointValue */

/**
 * Get field point value at coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
FieldPointValue *
Grid<TCoord>::getFieldPointValue (grid_coord coord) /**< index in grid */
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
} /* Grid<TCoord>::getFieldPointValue */

/**
 * Get field point value at relative coordinate in grid
 *
 * @return field point value
 */
template <class TCoord>
FieldPointValue *
Grid<TCoord>::getFieldPointValueByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  return getFieldPointValue (relPosition);
} /* Grid<TCoord>::getFieldPointValueByAbsolutePos */

/**
 * Get field point value at relative coordinate in grid or null
 *
 * @return field point value or null
 */
template <class TCoord>
FieldPointValue *
Grid<TCoord>::getFieldPointValueOrNullByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  return getFieldPointValue (relPosition);
} /* Grid<TCoord>::getFieldPointValueOrNullByAbsolutePos */

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
  return pos;
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
  return pos;
} /* gGrid<TCoord>::getRelativePosition */

/**
 * Get name of grid
 *
 * @return name of grid
 */
template <class TCoord>
const std::string &
Grid<TCoord>::getName () const
{
  return gridName;
} /* Grid<TCoord>::getName */

/**
 * Initialize grid field values with default values
 */
template <class TCoord>
void
Grid<TCoord>::initialize ()
{
  for (grid_coord i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = new FieldPointValue ();
  }
} /* Grid<TCoord>::initialize */

/**
 * Initialize grid field values with default values
 */
template <class TCoord>
void
Grid<TCoord>::initialize (FieldValue cur)
{
  for (grid_coord i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = new FieldPointValue ();
    gridValues[i]->setCurValue (cur);
  }
} /* Grid<TCoord>::initialize */

#endif /* GRID_H */

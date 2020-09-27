#ifndef GRID_H
#define GRID_H

#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>

#include "Assert.h"
#include "FieldValue.h"
#include "GridCoordinate3D.h"
#include "Settings.h"

/**
 * Storage of FieldValue in N-dimensional mode
 */
template <class TCoord>
class VectorFieldValues
{
}; /* VectorFieldValues */

/**
 * Storage of FieldValue in 1-dimensional mode
 */
template <>
class VectorFieldValues<GridCoordinate1D>
{
  /**
   * Size of vector
   */
  GridCoordinate1D size;

  /**
   * Pointer to raw data
   */
  FieldValue *data;

private:

  /**
   * Allocate raw buffer
   */
  void alloc ()
  {
    ASSERT (data == NULLPTR);
    data = new FieldValue[size.get1 ()];
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      data[i] = FIELDVALUE (0, 0);
    }
  } /* alloc */

  /**
   * Free raw buffer
   */
  void free ()
  {
    ASSERT (data != NULLPTR);
    delete[] data;
    data = NULLPTR;
  } /* free */

public:

  /**
   * Iterator
   */
  class Iterator
  {
    /**
     * Current iterator coordinate
     * Values from start to end allowed, including both (end is used for grid->end())
     */
    GridCoordinate1D pos;

    /**
     * Start iterator coordinate
     */
    GridCoordinate1D start;

    /**
     * End iterator coordinate
     */
    GridCoordinate1D end;

  public:

    /**
     * Constructor
     */
    Iterator (const GridCoordinate1D & new_pos, /**< current coordinate */
              const GridCoordinate1D & new_start, /**< start coordinate */
              const GridCoordinate1D & new_end) /**< end coordinate */
    : pos (new_pos)
    , start (new_start)
    , end (new_end)
    {
      ASSERT (pos >= start)
      ASSERT (pos <= end);
    } /* Iterator */

    /**
     * Pre-increment operator
     *
     * @return iterator
     */
    Iterator & operator++ ()
    {
      ASSERT (pos < end);
      pos.set1 (pos.get1 () + 1);
      return *this;
    } /* operator++ */

    /**
     * Pre-decrement operator
     *
     * @return iterator
     */
    Iterator & operator-- ()
    {
      ASSERT (pos >= start);
      pos.set1 (pos.get1 () - 1);
      return *this;
    } /* operator-- */

    /**
     * Equality comparison
     *
     * @return result of comparison
     */
    bool operator== (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos == rhs.pos;
    } /* operator== */

    /**
     * Non-equality comparison
     *
     * @return result of comparison
     */
    bool operator!= (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos != rhs.pos;
    } /* operator!= */

    /**
     * Get current coordinate of iterator
     *
     * @return current coordinate
     */
    const GridCoordinate1D & getPos () const
    {
      ASSERT (pos >= start);
      ASSERT (pos < end);
      return pos;
    } /* getPos */

    /**
     * Get end iterator for specified start and end coordinates
     *
     * @return end iterator
     */
    static Iterator getEndIterator (const GridCoordinate1D &start, /**< start coordinate */
                                    const GridCoordinate1D &end) /**< end coordinate */
    {
      return Iterator (GRID_COORDINATE_1D (end.get1 (), end.getType1 ()), start, end);
    } /* getEndIterator */
  }; /* Iterator */

public:

  /**
   * Constructor
   */
  VectorFieldValues<GridCoordinate1D> (const GridCoordinate1D & new_size) /**< size of storage */
  : size (new_size)
  , data (NULLPTR)
  {
    alloc ();
  } /* VectorFieldValues */

  /**
   * Destructor
   */
  ~VectorFieldValues<GridCoordinate1D> ()
  {
    free ();
  } /* ~VectorFieldValues */

  /**
   * Get FieldValue at coordinate
   *
   * @return FieldValue
   */
  FieldValue * get(const GridCoordinate1D & coord) /**< coordinate */
  {
    return &data[coord.get1 ()];
  } /* get */

  /**
   * Set FieldValue at coordinate
   */
  void set (const GridCoordinate1D & coord, /**< coordinate */
            const FieldValue & val) /**< field value */
  {
    data[coord.get1 ()] = val;
  } /* set */

  /**
   * Get size of storage
   *
   * @return size of storage
   */
  const GridCoordinate1D & getSize () const
  {
    return size;
  } /* getSize */

  /**
   * Copy raw data
   */
  void copy (VectorFieldValues<GridCoordinate1D> *values) /**< raw values to copy */
  {
    ASSERT (size == values->size);
    memcpy (data, values->data, size.get1 () * sizeof (FieldValue));
  } /* copy */

  /**
   * Initialize field values with some value
   */
  void initialize (const FieldValue & val) /**< value to initialize all field values with */
  {
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      data[i] = val;
    }
  } /* initialize */

  /**
   * Resize raw storage
   */
  void resizeAndEmpty (const GridCoordinate1D &new_size) /**< new size */
  {
    free ();
    size = new_size;
    alloc ();
  } /* resizeAndEmpty */

  /**
   * Get begin iterator
   *
   * @return begin iterator
   */
  Iterator begin () const
  {
    return Iterator (size.getZero (), size.getZero (), size);
  } /* begin */

  /**
   * Get end iterator
   *
   * @return end iterator
   */
  Iterator end () const
  {
    return Iterator::getEndIterator (size.getZero (), size);
  } /* end */
}; /* VectorFieldValues */

/**
 * Storage of FieldValue in 2-dimensional mode
 */
template <>
class VectorFieldValues<GridCoordinate2D>
{
  /**
   * Size of vector
   */
  GridCoordinate2D size;

  /**
   * Pointer to raw data
   */
  FieldValue **data;

private:

  /**
   * Allocate raw buffer
   */
  void alloc ()
  {
    ASSERT (data == NULLPTR);
    data = new FieldValue *[size.get1 ()];
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      data[i] = new FieldValue[size.get2 ()];

      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        data[i][j] = FIELDVALUE (0, 0);
      }
    }
  } /* alloc */

  /**
   * Free raw buffer
   */
  void free ()
  {
    ASSERT (data != NULLPTR);
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      ASSERT (data[i] != NULLPTR);
      delete[] data[i];
      data[i] = NULLPTR;
    }
    delete[] data;
    data = NULLPTR;
  } /* free */

public:

  /**
   * Iterator
   */
  class Iterator
  {
    /**
     * Current iterator coordinate
     * Values from start to end allowed, including both (end is used for grid->end())
     */
    GridCoordinate2D pos;

    /**
     * Start iterator coordinate
     */
    GridCoordinate2D start;

    /**
     * End iterator coordinate
     */
    GridCoordinate2D end;

  public:

    /**
     * Constructor
     */
    Iterator (const GridCoordinate2D &new_pos, /**< current coordinate */
              const GridCoordinate2D &new_start, /**< start coordinate */
              const GridCoordinate2D &new_end) /**< end coordinate */
    : pos (new_pos)
    , start (new_start)
    , end (new_end)
    {
      ASSERT (pos >= start)
      ASSERT (pos <= end);
    } /* Iterator */

    /**
     * Pre-increment operator
     *
     * @return iterator
     */
    Iterator & operator++ ()
    {
      ASSERT (pos < end);
      if (pos.get2 () == end.get2 () - 1)
      {
        pos.set2 (start.get2 ());
        pos.set1 (pos.get1 () + 1);
      }
      else
      {
        pos.set2 (pos.get2 () + 1);
      }
      return *this;
    } /* operator++ */

    /**
     * Pre-decrement operator
     *
     * @return iterator
     */
    Iterator & operator-- ()
    {
      ASSERT (pos >= start);
      if (pos.get2 () == start.get2 ())
      {
        pos.set2 (end.get2 () - 1);
        pos.set1 (pos.get1 () - 1);
      }
      else
      {
        pos.set2 (pos.get2 () - 1);
      }
      return *this;
    } /* operator-- */

    /**
     * Equality comparison
     *
     * @return result of comparison
     */
    bool operator== (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos == rhs.pos;
    } /* operator== */

    /**
     * Non-equality comparison
     *
     * @return result of comparison
     */
    bool operator!= (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos != rhs.pos;
    } /* operator!= */

    /**
     * Get current coordinate of iterator
     *
     * @return current coordinate
     */
    const GridCoordinate2D & getPos () const
    {
      ASSERT (pos >= start);
      ASSERT (pos < end);
      return pos;
    } /* getPos */

    /**
     * Get end iterator for specified start and end coordinates
     *
     * @return end iterator
     */
    static Iterator getEndIterator (const GridCoordinate2D &start, /**< start coordinate */
                                    const GridCoordinate2D &end) /**< end coordinate */
    {
      return Iterator (GRID_COORDINATE_2D (end.get1 (), start.get2 (),
                                           end.getType1 (), end.getType2 ()),
                       start, end);
    } /* getEndIterator */
  }; /* Iterator */

public:

  /**
   * Constructor
   */
  VectorFieldValues<GridCoordinate2D> (const GridCoordinate2D & new_size) /**< size of storage */
  : size (new_size)
  , data (NULLPTR)
  {
    alloc ();
  } /* VectorFieldValues */

  /**
   * Destructor
   */
  ~VectorFieldValues<GridCoordinate2D> ()
  {
    free ();
  } /* ~VectorFieldValues */

  /**
   * Get FieldValue at coordinate
   *
   * @return FieldValue
   */
  FieldValue * get(const GridCoordinate2D & coord) /**< coordinate */
  {
    return &data[coord.get1 ()][coord.get2 ()];
  } /* get */

  /**
   * Set FieldValue at coordinate
   */
  void set (const GridCoordinate2D & coord, /**< coordinate */
            const FieldValue & val) /**< field value */
  {
    data[coord.get1 ()][coord.get2 ()] = val;
  } /* set */

  /**
   * Get size of storage
   *
   * @return size of storage
   */
  const GridCoordinate2D & getSize () const
  {
    return size;
  } /* getSize */

  /**
   * Copy raw data
   */
  void copy (VectorFieldValues<GridCoordinate2D> *values) /**< raw values to copy */
  {
    ASSERT (size == values->size);
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      memcpy (data[i], values->data[i], size.get2 () * sizeof (FieldValue));
    }
  } /* copy */

  /**
   * Initialize field values with some value
   */
  void initialize (const FieldValue & val) /**< value to initialize all field values with */
  {
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        data[i][j] = val;
      }
    }
  } /* initialize */

  /**
   * Resize raw storage
   */
  void resizeAndEmpty (const GridCoordinate2D &new_size) /**< new size */
  {
    free ();
    size = new_size;
    alloc ();
  } /* resizeAndEmpty */

  /**
   * Get begin iterator
   *
   * @return begin iterator
   */
  Iterator begin () const
  {
    return Iterator (size.getZero (), size.getZero (), size);
  } /* begin */

  /**
   * Get end iterator
   *
   * @return end iterator
   */
  Iterator end () const
  {
    return Iterator::getEndIterator (size.getZero (), size);
  } /* end */
}; /* VectorFieldValues */

/**
 * Storage of FieldValue in 2-dimensional mode
 */
template <>
class VectorFieldValues<GridCoordinate3D>
{
  /**
   * Size of vector
   */
  GridCoordinate3D size;

  /**
   * Pointer to raw data
   */
  FieldValue ***data;

private:

  /**
   * Allocate raw buffer
   */
  void alloc ()
  {
    ASSERT (data == NULLPTR);
    data = new FieldValue **[size.get1 ()];
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      data[i] = new FieldValue *[size.get2 ()];
      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        data[i][j] = new FieldValue [size.get3 ()];

        for (grid_coord k = 0; k < size.get3 (); ++k)
        {
          data[i][j][k] = FIELDVALUE (0, 0);
        }
      }
    }
  } /* alloc */

  /**
   * Free raw buffer
   */
  void free ()
  {
    ASSERT (data != NULLPTR);
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      ASSERT (data[i] != NULLPTR);
      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        ASSERT (data[i][j] != NULLPTR);
        delete[] data[i][j];
        data[i][j] = NULLPTR;
      }
      delete[] data[i];
      data[i] = NULLPTR;
    }
    delete[] data;
    data = NULLPTR;
  } /* free */

public:

  /**
   * Iterator
   */
  class Iterator
  {
    /**
     * Current iterator coordinate
     * Values from start to end allowed, including both (end is used for grid->end())
     */
    GridCoordinate3D pos;

    /**
     * Start iterator coordinate
     */
    GridCoordinate3D start;

    /**
     * End iterator coordinate
     */
    GridCoordinate3D end;

  public:

    /**
     * Constructor
     */
    Iterator (const GridCoordinate3D &new_pos, /**< current coordinate */
              const GridCoordinate3D &new_start, /**< start coordinate */
              const GridCoordinate3D &new_end) /**< end coordinate */
    : pos (new_pos)
    , start (new_start)
    , end (new_end)
    {
      ASSERT (pos >= start)
      ASSERT (pos <= end);
    } /* Iterator */

    /**
     * Pre-increment operator
     *
     * @return iterator
     */
    Iterator & operator++ ()
    {
      ASSERT (pos < end);
      if (pos.get3 () == end.get3 () - 1)
      {
        if (pos.get2 () == end.get2 () - 1)
        {
          pos.set3 (start.get3 ());
          pos.set2 (start.get2 ());
          pos.set1 (pos.get1 () + 1);
        }
        else
        {
          pos.set3 (start.get3 ());
          pos.set2 (pos.get2 () + 1);
        }
      }
      else
      {
        pos.set3 (pos.get3 () + 1);
      }
      return *this;
    } /* operator++ */

    /**
     * Pre-decrement operator
     *
     * @return iterator
     */
    Iterator & operator-- ()
    {
      ASSERT (pos >= start);
      if (pos.get3 () == start.get3 ())
      {
        if (pos.get2 () == start.get2 ())
        {
          pos.set3 (end.get3 () - 1);
          pos.set2 (end.get2 () - 1);
          pos.set1 (pos.get1 () - 1);
        }
        else
        {
          pos.set3 (end.get3 () - 1);
          pos.set2 (pos.get2 () - 1);
        }
      }
      else
      {
        pos.set3 (pos.get3 () - 1);
      }
      return *this;
    } /* operator-- */

    /**
     * Equality comparison
     *
     * @return result of comparison
     */
    bool operator== (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos == rhs.pos;
    } /* operator== */

    /**
     * Non-equality comparison
     *
     * @return result of comparison
     */
    bool operator!= (const Iterator &rhs) const /**< iterator to compare to */
    {
      ASSERT (start == rhs.start);
      ASSERT (end == rhs.end);
      return pos != rhs.pos;
    } /* operator!= */

    /**
     * Get current coordinate of iterator
     *
     * @return current coordinate
     */
    const GridCoordinate3D & getPos () const
    {
      ASSERT (pos >= start);
      ASSERT (pos < end);
      return pos;
    } /* getPos */

    /**
     * Get end iterator for specified start and end coordinates
     *
     * @return end iterator
     */
    static Iterator getEndIterator (const GridCoordinate3D &start, /**< start coordinate */
                                    const GridCoordinate3D &end) /**< end coordinate */
    {
      return Iterator (GRID_COORDINATE_3D (end.get1 (), start.get2 (), start.get3 (),
                                           end.getType1 (), end.getType2 (), end.getType3 ()),
                       start, end);
    } /* getEndIterator */
  }; /* Iterator */

public:

  /**
   * Constructor
   */
  VectorFieldValues<GridCoordinate3D> (const GridCoordinate3D & new_size) /**< size of storage */
  : size (new_size)
  , data (NULLPTR)
  {
    alloc ();
  } /* VectorFieldValues */

  /**
   * Destructor
   */
  ~VectorFieldValues<GridCoordinate3D> ()
  {
    free ();
  } /* ~VectorFieldValues */

  /**
   * Get FieldValue at coordinate
   *
   * @return FieldValue
   */
  FieldValue * get(const GridCoordinate3D & coord) /**< coordinate */
  {
    return &data[coord.get1 ()][coord.get2 ()][coord.get3 ()];
  } /* get */

  /**
   * Set FieldValue at coordinate
   */
  void set (const GridCoordinate3D & coord, /**< coordinate */
            const FieldValue & val) /**< field value */
  {
    data[coord.get1 ()][coord.get2 ()][coord.get3 ()] = val;
  } /* set */

  /**
   * Get size of storage
   *
   * @return size of storage
   */
  const GridCoordinate3D & getSize () const
  {
    return size;
  } /* getSize */

  /**
   * Copy raw data
   */
  void copy (VectorFieldValues<GridCoordinate3D> *values) /**< raw values to copy */
  {
    ASSERT (size == values->size);
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        memcpy (data[i][j], values->data[i][j], size.get3 () * sizeof (FieldValue));
      }
    }
  } /* copy */

  /**
   * Initialize field values with some value
   */
  void initialize (const FieldValue & val) /**< value to initialize all field values with */
  {
    for (grid_coord i = 0; i < size.get1 (); ++i)
    {
      for (grid_coord j = 0; j < size.get2 (); ++j)
      {
        for (grid_coord k = 0; k < size.get3 (); ++k)
        {
          data[i][j][k] = val;
        }
      }
    }
  } /* initialize */

  /**
   * Resize raw storage
   */
  void resizeAndEmpty (const GridCoordinate3D &new_size) /**< new size */
  {
    free ();
    size = new_size;
    alloc ();
  } /* resizeAndEmpty */

  /**
   * Get begin iterator
   *
   * @return begin iterator
   */
  Iterator begin () const
  {
    return Iterator (size.getZero (), size.getZero (), size);
  } /* begin */

  /**
   * Get end iterator
   *
   * @return end iterator
   */
  Iterator end () const
  {
    return Iterator::getEndIterator (size.getZero (), size);
  } /* end */
}; /* VectorFieldValues */

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
   * Vector of points in grid.
   */
  std::vector<VectorFieldValues<TCoord> *> gridValues;

  /**
   * Name of the grid.
   */
  std::string gridName;

  /*
   * TODO: add debug uninitialized flag
   */

protected:

  static bool isLegitIndex (const TCoord &, const TCoord &);

private:

  Grid<TCoord> & operator = (const Grid<TCoord> &);
  Grid (const Grid<TCoord> &);

protected:

  bool isLegitIndex (const TCoord &) const;

public:

  Grid (const TCoord&, int, const char * = "unnamed");
  Grid (int, const char * = "unnamed");
  virtual ~Grid ();

  const TCoord & getSize () const;
  virtual TCoord getTotalPosition (const TCoord &) const;
  virtual TCoord getTotalSize () const;
  virtual TCoord getRelativePosition (const TCoord &) const;
  virtual TCoord getComputationStart (const TCoord &) const;
  virtual TCoord getComputationEnd (const TCoord &) const;

  void setFieldValue (const FieldValue &, const TCoord &, int);
  FieldValue * getFieldValue (const TCoord &, int);

  virtual FieldValue * getFieldValueByAbsolutePos (const TCoord &, int);
  virtual FieldValue * getFieldValueOrNullByAbsolutePos (const TCoord &, int);

  virtual FieldValue * getFieldValueCurrentAfterShiftByAbsolutePos (const TCoord &);
  virtual FieldValue * getFieldValueOrNullCurrentAfterShiftByAbsolutePos (const TCoord &);
  virtual FieldValue * getFieldValuePreviousAfterShiftByAbsolutePos (const TCoord &);
  virtual FieldValue * getFieldValueOrNullPreviousAfterShiftByAbsolutePos (const TCoord &);

  virtual TCoord getChunkStartPosition () const
  {
    return getSize ().getZero ();
  }

  virtual bool isBufferLeftPosition (const TCoord & pos) const
  {
    return false;
  }

  virtual bool isBufferRightPosition (const TCoord & pos) const
  {
    return false;
  }

  void shiftInTime ();

  const char * getName () const;

  void initialize (const FieldValue &);

  VectorFieldValues<TCoord> * getRaw (int);
  typename VectorFieldValues<TCoord>::Iterator begin ();
  typename VectorFieldValues<TCoord>::Iterator end ();

  int getCountStoredSteps () const
  {
    return gridValues.size ();
  }

  void copy (const Grid<TCoord> *grid)
  {
    ASSERT (size == grid->size);
    ASSERT (gridValues.size () == grid->gridValues.size ());

    for (int i = 0; i < gridValues.size (); ++i)
    {
      gridValues[i]->copy (grid->gridValues[i]);
    }
  }
}; /* Grid */

/**
 * Constructor of grid
 */
template <class TCoord>
Grid<TCoord>::Grid (const TCoord &s, /**< size of grid */
                    int storedSteps, /**< number of steps in time for which to store grid values */
                    const char *name) /**< name of grid */
  : size (s)
  , gridValues (storedSteps)
  , gridName (name)
{
  ASSERT (storedSteps > 0);

  for (int i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = new VectorFieldValues<TCoord> (size);
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' with %" PRIu32 " stored steps and raw size: %" PRIu64 ".\n",
    gridName.data (), (uint32_t) gridValues.size (), (uint64_t)size.calculateTotalCoord ());
} /* Grid<TCoord>::Grid */

/**
 * Constructor of grid without size
 */
template <class TCoord>
Grid<TCoord>::Grid (int storedSteps, /**< number of steps in time for which to store grid values */
                    const char *name) /**< name of grid */
  : gridValues (storedSteps)
  , gridName (name)
{
  ASSERT (storedSteps > 0);

  for (int i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = NULLPTR;
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' with %lu stored steps without size.\n",
    gridName.data (), gridValues.size ());
} /* Grid<TCoord>::Grid */

/**
 * Destructor of grid. Should delete all field point values
 */
template <class TCoord>
Grid<TCoord>::~Grid ()
{
  for (int i = 0; i < gridValues.size (); ++i)
  {
    delete gridValues[i];
    gridValues[i] = NULLPTR;
  }
} /* Grid<TCoord>::~Grid */


/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
 template <class TCoord>
 bool
 Grid<TCoord>::isLegitIndex (const TCoord &position, /**< coordinate in grid */
                             const TCoord &sizeCoord) /**< size of grid */
{
  return position < sizeCoord;
} /* Grid<TCoord>::isLegitIndex */

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
 * Get size of the grid
 *
 * @return size of the grid
 */
template <class TCoord>
const TCoord &
Grid<TCoord>::getSize () const
{
  return size;
} /* Grid<TCoord>::getSize */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <class TCoord>
TCoord
Grid<TCoord>::getComputationStart (const TCoord &diffPosStart) const /**< offset from the left border */
{
  return getSize ().getZero () + diffPosStart;
} /* Grid<TCoord>::getComputationStart */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
template <class TCoord>
TCoord
Grid<TCoord>::getComputationEnd (const TCoord & diffPosEnd) const /**< offset from the right border */
{
  return getSize () - diffPosEnd;
} /* Grid<TCoord>::getComputationEnd () */

/**
 * Set field value at coordinate in grid
 */
template <class TCoord>
void
Grid<TCoord>::setFieldValue (const FieldValue & value, /**< field point value */
                             const TCoord & position, /**< coordinate in grid */
                             int time_step_back) /**< index of previous time step, starting from current (0) */
{
  ASSERT (isLegitIndex (position));
  ASSERT (time_step_back < gridValues.size ());

  gridValues[time_step_back]->set (position, value);
} /* Grid<TCoord>::setFieldValue */

/**
 * Get field value at coordinate in grid
 *
 * @return field value
 */
template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValue (const TCoord &position, /**< coordinate in grid */
                             int time_step_back) /**< index of previous time step, starting from current (0) */
{
  ASSERT (isLegitIndex (position));
  ASSERT (time_step_back < gridValues.size ());

  return gridValues[time_step_back]->get (position);
} /* Grid<TCoord>::getFieldValue */

/**
 * Get field value at relative coordinate in grid
 *
 * @return field value
 */
template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValueByAbsolutePos (const TCoord &relPosition, /**< relative coordinate in grid */
                                          int time_step_back) /**< index of previous time step, starting from current (0) */
{
  return getFieldValue (relPosition, time_step_back);
} /* Grid<TCoord>::getFieldValueByAbsolutePos */

/**
 * Get field value at relative coordinate in grid or null
 *
 * @return field value or null
 */
template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValueOrNullByAbsolutePos (const TCoord &relPosition, /**< relative coordinate in grid */
                                                int time_step_back) /**< index of previous time step, starting from current (0) */
{
  return getFieldValueByAbsolutePos (relPosition, time_step_back);
} /* Grid<TCoord>::getFieldValueOrNullByAbsolutePos */

template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValueCurrentAfterShiftByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  return getFieldValue (relPosition, 1);
}

template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValueOrNullCurrentAfterShiftByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  return getFieldValueCurrentAfterShiftByAbsolutePos (relPosition);
}

template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValuePreviousAfterShiftByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  if (gridValues.size () > 2)
  {
    return getFieldValue (relPosition, 2);
  }
  else
  {
    return getFieldValue (relPosition, 0);
  }
}

template <class TCoord>
FieldValue *
Grid<TCoord>::getFieldValueOrNullPreviousAfterShiftByAbsolutePos (const TCoord &relPosition) /**< relative coordinate in grid */
{
  return getFieldValuePreviousAfterShiftByAbsolutePos (relPosition);
}

/**
 * Get total position in grid. Is equal to position in non-parallel grid
 *
 * @return total position in grid
 */
template <class TCoord>
TCoord
Grid<TCoord>::getTotalPosition (const TCoord & pos) const /**< position in grid */
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
Grid<TCoord>::getRelativePosition (const TCoord & pos) const /**< position in grid */
{
  return pos;
} /* gGrid<TCoord>::getRelativePosition */

/**
 * Get name of grid
 *
 * @return name of grid
 */
template <class TCoord>
const char *
Grid<TCoord>::getName () const
{
  return gridName.c_str ();
} /* Grid<TCoord>::getName */

/**
 * Initialize current grid field values with default values
 */
template <class TCoord>
void
Grid<TCoord>::initialize (const FieldValue & cur)
{
  ASSERT (gridValues.size () > 0);
  gridValues[0]->initialize (cur);
} /* Grid<TCoord>::initialize */

template <class TCoord>
VectorFieldValues<TCoord> *
Grid<TCoord>::getRaw (int time_step_back)
{
  ASSERT (time_step_back < gridValues.size ());

  return gridValues[time_step_back];
}

template <class TCoord>
typename VectorFieldValues<TCoord>::Iterator
Grid<TCoord>::begin ()
{
  return getRaw (0)->begin ();
}

template <class TCoord>
typename VectorFieldValues<TCoord>::Iterator
Grid<TCoord>::end ()
{
  return getRaw (0)->end ();
}

/**
 * Replace previous time layer with current and so on
 */
template <class TCoord>
void
Grid<TCoord>::shiftInTime ()
{
  /*
   * Reuse oldest grid as new current
   */
  ASSERT (gridValues.size () > 0);

  VectorFieldValues<TCoord> *oldest = gridValues[gridValues.size () - 1];

  for (int i = gridValues.size () - 1; i >= 1; --i)
  {
    gridValues[i] = gridValues[i - 1];
  }

  gridValues[0] = oldest;
} /* Grid<TCoord>::shiftInTime */

#endif /* GRID_H */

#ifndef GRID_2D_H
#define GRID_2D_H

#include "GridCoordinate2D.h"
#include "Grid.h"

class Grid2D: public Grid
{
  // Size of the grid.
  // For parallel grid - size of current node plus size of buffers.
  GridCoordinate2D size;

  // Vector of points in grid.
  // Owns this. Deletes all FieldPointValue* itself.
  VectorFieldPointValues gridValues;

  // Current time step.
  uint32_t timeStep;

private:

  // Check whether position is appropriate to get/set value from.
  static bool isLegitIndex (const GridCoordinate2D& position,
                            const GridCoordinate2D& sizeCoord);

  // Calculate N-dimensional coordinate from position.
  static grid_iter calculateIndexFromPosition (const GridCoordinate2D& position,
                                               const GridCoordinate2D& sizeCoord);

private:

  // Get values in the grid.
  VectorFieldPointValues& getValues ();

  // Replace previous layer with current and so on.
  void shiftInTime ();

  // Check whether position is appropriate to get/set value from.
  bool isLegitIndex (const GridCoordinate2D& position) const;

  // Calculate N-dimensional coordinate from position.
  grid_iter calculateIndexFromPosition (const GridCoordinate2D& position) const;

public:

  Grid2D (const GridCoordinate2D& s, uint32_t step);
  ~Grid2D ();

  // Get size of the grid.
  const GridCoordinate2D& getSize () const;

  // Calculate position from three-dimensional coordinate.
  GridCoordinate2D calculatePositionFromIndex (grid_iter index) const;

  // Set field point at coordinate in grid.
  void setFieldPointValue (FieldPointValue* value, const GridCoordinate2D& position);

  void setFieldPointValueCurrent (const FieldValue& value,
                                  const GridCoordinate2D& position);

  // Get field point at coordinate in grid.
  FieldPointValue* getFieldPointValue (const GridCoordinate2D& position);
  FieldPointValue* getFieldPointValue (grid_iter coord);

  // Switch to next time step.
  virtual void nextTimeStep () override;
};

#endif /* GRID_2D_H */

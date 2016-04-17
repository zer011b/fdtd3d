#ifndef GRID_3D_H
#define GRID_3D_H

#include "GridCoordinate3D.h"
#include "Grid.h"

class Grid3D: public Grid
{
  // Size of the grid.
  // For parallel grid - size of current node plus size of buffers.
  GridCoordinate3D size;

  // Vector of points in grid.
  // Owns this. Deletes all FieldPointValue* itself.
  VectorFieldPointValues gridValues;

  // Current time step.
  uint32_t timeStep;

private:

  // Check whether position is appropriate to get/set value from.
  static bool isLegitIndex (const GridCoordinate3D& position,
                            const GridCoordinate3D& sizeCoord);

  // Calculate N-dimensional coordinate from position.
  static grid_iter calculateIndexFromPosition (const GridCoordinate3D& position,
                                               const GridCoordinate3D& sizeCoord);

private:

  // Get values in the grid.
  VectorFieldPointValues& getValues ();

  // Replace previous layer with current and so on.
  void shiftInTime ();

  // Check whether position is appropriate to get/set value from.
  bool isLegitIndex (const GridCoordinate3D& position) const;

  // Calculate N-dimensional coordinate from position.
  grid_iter calculateIndexFromPosition (const GridCoordinate3D& position) const;

public:

  Grid3D (const GridCoordinate3D& s, uint32_t step);
  ~Grid3D ();

  // Get size of the grid.
  const GridCoordinate3D& getSize () const;

  // Calculate position from three-dimensional coordinate.
  GridCoordinate3D calculatePositionFromIndex (grid_iter index) const;

  // Set field point at coordinate in grid.
  void setFieldPointValue (FieldPointValue* value, const GridCoordinate3D& position);

  void setFieldPointValueCurrent (const FieldValue& value,
                                  const GridCoordinate3D& position);

  // Get field point at coordinate in grid.
  FieldPointValue* getFieldPointValue (const GridCoordinate3D& position);
  FieldPointValue* getFieldPointValue (grid_iter coord);

  // Switch to next time step.
  virtual void nextTimeStep () override;
};

#endif /* GRID_3D_H */

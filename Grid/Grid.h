#ifndef GRID_H
#define GRID_H

#include <vector>

#include "FieldPoint.h"

// Vector of points in grid.
typedef std::vector<FieldPointValue*> VectorFieldPointValues;

// Grid interface.
class Grid
{
public:

  virtual ~Grid () {}

  // Switch to next time step.
  virtual void nextTimeStep () = 0;
};


#endif /* GRID_H */

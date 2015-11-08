#ifndef LOADER_H
#define LOADER_H

#include "Commons.h"

// Basic class for all loaders.
class Loader: public GridFileManager
{
protected:

  // Maximum positive value in grid.
  FieldPointValue maxValuePos;
  // Maximum negative value in grid.
  FieldPointValue maxValueNeg;

public:

  virtual void LoadGrid (Grid& grid) const = 0;

  // Setter and getter for maximum positive value.
  void setMaxValuePos (FieldPointValue& value)
  {
    maxValuePos = value;
  }
  const FieldPointValue& getMaxValuePos () const
  {
    return maxValuePos;
  }

  // Setter and getter for maximum negative value.
  void setMaxValueNeg (FieldPointValue& value)
  {
    maxValueNeg = value;
  }
  const FieldPointValue& getMaxValueNeg () const
  {
    return maxValueNeg;
  }
};

#endif /* LOADER_H */

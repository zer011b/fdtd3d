#ifndef LOADER_H
#define LOADER_H

#include "commons.h"

class Loader: public GridFileManager
{
protected:

  FieldPointValue maxValuePos;
  FieldPointValue maxValueNeg;

public:

  virtual void LoadGrid (Grid& grid) const = 0;

  void setMaxValuePos (FieldPointValue& value)
  {
    maxValuePos = value;
  }
  const FieldPointValue& getMaxValuePos () const
  {
    return maxValuePos;
  }
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

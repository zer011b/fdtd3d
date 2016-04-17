#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include "Loader.h"
#include "BMPHelper.h"

// Grid loader from BMP files.
template <class TGrid>
class BMPLoader: public Loader<TGrid>
{
  // Maximum positive value in grid.
  FieldPointValue maxValuePos;
  // Maximum negative value in grid.
  FieldPointValue maxValueNeg;

  static BMPHelper BMPhelper;

private:

  static void loadFromFile (TGrid &grid, GridFileType load_type);
  static void loadFromFile (TGrid &grid);

public:

  virtual ~BMPLoader () {}

  // Function to call for every grid type.
  virtual void loadGrid (TGrid &grid) const override;

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

#endif /* BMP_LOADER_H */

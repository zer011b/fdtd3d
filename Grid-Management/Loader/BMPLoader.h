#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include "Loader.h"
#include "EasyBMP.h"

// Grid loader from BMP files.
class BMPLoader: public Loader
{
  // Maximum positive value in grid.
  FieldPointValue maxValuePos;
  // Maximum negative value in grid.
  FieldPointValue maxValueNeg;

#if defined (GRID_1D)
  // Load one-dimensional grid.
  void load1D (Grid& grid) const;
#else
#if defined (GRID_2D)
  // Load two-dimensional grid.
  void load2D (Grid& grid) const;
#else
#if defined (GRID_3D)
  // Load three-dimensional grid.
  void load3D (Grid& grid) const;
#endif
#endif
#endif

  // Return pixel with colors according to values.
  FieldValue getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                                const FieldValue& max) const;

  // Load flat grids: 1D and 2D.
  void loadFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const;

  void loadFromFile (Grid& grid, const grid_iter& sx, const grid_iter& sy, GridFileType load_type) const;

public:

  // Function to call for every grid type.
  void loadGrid (Grid& grid) const override;

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

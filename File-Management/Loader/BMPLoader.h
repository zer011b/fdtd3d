#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include "BMPHelper.h"
#include "Loader.h"

/**
 * Grid loader from BMP files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class BMPLoader: public Loader<TCoord>
{
  // Maximum positive value in grid.
  FieldPointValue maxValuePos;

  // Maximum negative value in grid.
  FieldPointValue maxValueNeg;

  // Helper class for usage with BMP files.
  static BMPHelper BMPhelper;

private:

  // Load grid from file for specific layer.
  void loadFromFile (Grid<TCoord> &grid, GridFileType load_type) const;

  // Load grid from file for all layers.
  void loadFromFile (Grid<TCoord> &grid) const;

public:

  virtual ~BMPLoader () {}

  // Virtual method for grid loading.
#ifdef CXX11_ENABLED
  virtual void loadGrid (Grid<TCoord> &grid) const override;
#else
  virtual void loadGrid (Grid<TCoord> &grid) const;
#endif

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

/**
 * ======== Template implementation ========
 */

/**
 * Load grid from file for all layers.
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadFromFile (Grid<TCoord> &grid) const
{
  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
#ifdef CXX11_ENABLED
  if (GridFileManager::type == ALL)
#else
  if (this->GridFileManager::type == ALL)
#endif
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

template <class TCoord>
BMPHelper BMPLoader<TCoord>::BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED);

#endif /* BMP_LOADER_H */

#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include "BMPHelper.h"
#include "Loader.h"

/**
 * Grid loader from BMP files.
 * Template class with coordinate parameter.
 *
 * NOTE: ".bmp" dumper/loader can't reproduce field values precisely.
 *       Consequent dump to and load from ".bmp" file of grid will change grid values.
 */
template <class TCoord>
class BMPLoader: public Loader<TCoord>
{
  // Helper class for usage with BMP files.
  static BMPHelper BMPhelper;

private:

  // Load grid from file for specific layer.
  void loadFromFile (Grid<TCoord> *grid, GridFileType load_type) const;

  // Load grid from file for all layers.
  void loadFromFile (Grid<TCoord> *grid) const;

public:

  virtual ~BMPLoader () {}

  // Virtual method for grid loading.
  virtual void loadGrid (Grid<TCoord> *grid) const CXX11_OVERRIDE;

  void initializeHelper (PaletteType colorPalette, OrthogonalAxis orthAxis)
  {
    BMPhelper.initialize (colorPalette, orthAxis);
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
BMPLoader<TCoord>::loadFromFile (Grid<TCoord> *grid) const
{
  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (this->GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

template <class TCoord>
BMPHelper BMPLoader<TCoord>::BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED, OrthogonalAxis::Z);

#endif /* BMP_LOADER_H */

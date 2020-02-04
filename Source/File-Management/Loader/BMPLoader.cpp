#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>

#include "BMPLoader.h"

/**
 * Load grid from file for specific layer for 1D.
 */
template<>
void
BMPLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid, /**< grid to load */
                                           GridCoordinate1D startCoord, /**< start loading from this coordinate */
                                           GridCoordinate1D endCoord, /**< end loading at this coordinate */
                                           int time_step_back) /**< relative time step at which to load */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > endCoord.getZero () && endCoord <= grid->getSize ());

  grid_coord sx = grid->getSize ().get1 ();
  grid_coord sy = 1;

  std::string imageReName;
  std::string imageReNameTxt;
  std::string imageImName;
  std::string imageImNameTxt;
  std::string imageModName;
  std::string imageModNameTxt;
  setupNames (imageReName, imageReNameTxt, imageImName, imageImNameTxt, imageModName, imageModNameTxt, time_step_back, -1);

  // Create image for values and max/min values.
  BMP imageRe;
  BMP imageIm;

  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);
  imageRe.ReadFromFile (imageReName.c_str());

#ifdef COMPLEX_FIELD_VALUES
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
  imageIm.ReadFromFile (imageImName.c_str());
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe;
  FPValue minRe;
  FPValue maxIm;
  FPValue minIm;

  loadTxtFromFile (maxRe, minRe, maxIm, minIm, imageReNameTxt, imageImNameTxt);

  // Go through all values and set them.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());

    // Pixel coordinate.
    grid_coord px = pos.get1 ();
    grid_coord py = 0;

    loadPixel (px, py, grid, pos, time_step_back, minRe, maxRe, minIm, maxIm,
              imageRe, imageIm);
  }
} /* BMPLoader::loadFromFile */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Load grid from file for specific layer for 2D.
 */
template<>
void
BMPLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> *grid, /**< grid to load */
                                           GridCoordinate2D startCoord, /**< start loading from this coordinate */
                                           GridCoordinate2D endCoord, /**< end loading at this coordinate */
                                           int time_step_back) /**< relative time step at which to load */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && endCoord <= grid->getSize ());

  grid_coord sx = grid->getSize ().get1 ();
  grid_coord sy = grid->getSize ().get2 ();

  std::string imageReName;
  std::string imageReNameTxt;
  std::string imageImName;
  std::string imageImNameTxt;
  std::string imageModName;
  std::string imageModNameTxt;
  setupNames (imageReName, imageReNameTxt, imageImName, imageImNameTxt, imageModName, imageModNameTxt, time_step_back, -1);

  // Create image for values and max/min values.
  BMP imageRe;
  BMP imageIm;

  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);
  imageRe.ReadFromFile (imageReName.c_str());

#ifdef COMPLEX_FIELD_VALUES
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
  imageIm.ReadFromFile (imageImName.c_str());
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe;
  FPValue minRe;
  FPValue maxIm;
  FPValue minIm;

  loadTxtFromFile (maxRe, minRe, maxIm, minIm, imageReNameTxt, imageImNameTxt);

  // Go through all values and set them.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());

      // Pixel coordinate.
      grid_coord px = pos.get1 ();
      grid_coord py = pos.get2 ();

      loadPixel (px, py, grid, pos, time_step_back, minRe, maxRe, minIm, maxIm,
                imageRe, imageIm);
    }
  }
} /* BMPLoader::loadFromFile */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Load grid from file for specific layer for 3D.
 */
template<>
void
BMPLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> *grid, /**< grid to load */
                                           GridCoordinate3D startCoord, /**< start loading from this coordinate */
                                           GridCoordinate3D endCoord, /**< end loading at this coordinate */
                                           int time_step_back) /**< relative time step at which to load */
{
  ASSERT_MESSAGE ("3D loader is not implemented as it is considered unneeded (see Docs/Input-Output.md).")
} /* BMPLoader::loadFromFile */

#endif /* MODE_DIM3 */

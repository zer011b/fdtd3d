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
BMPLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid,
                                           GridCoordinate1D startCoord,
                                           GridCoordinate1D endCoord,
                                           int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_1D (0, startCoord.getType1 ()) && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_1D (0, startCoord.getType1 ()) && endCoord <= grid->getSize ());

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
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);
  imageRe.ReadFromFile (imageReName.c_str());

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
  imageIm.ReadFromFile (imageImName.c_str());
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe = 0;
  FPValue maxPosRe = 0;
  FPValue maxNegRe = 0;
  std::ifstream fileMaxRe;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxIm = 0;
  FPValue maxPosIm = 0;
  FPValue maxNegIm = 0;
  std::ifstream fileMaxIm;
#endif /* COMPLEX_FIELD_VALUES */

  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosRe >> maxNegRe;
  fileMaxRe.close();
  maxRe = maxPosRe - maxNegRe;

#ifdef COMPLEX_FIELD_VALUES
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosIm >> maxNegIm;
  fileMaxIm.close();
  maxIm = maxPosIm - maxNegIm;
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set them.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());
    grid_coord coord = grid->calculateIndexFromPosition (pos);

    // Pixel coordinate.
    grid_coord px = pos.get1 ();
    grid_coord py = 0;

    RGBApixel pixelRe = imageRe.GetPixel(px, py);
    FPValue currentValRe = BMPhelper.getValueFromPixel (pixelRe, maxNegRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
    RGBApixel pixelIm = imageIm.GetPixel(px, py);
    FPValue currentValIm = BMPhelper.getValueFromPixel (pixelIm, maxNegIm, maxIm);
#endif /* COMPLEX_FIELD_VALUES */

#ifdef COMPLEX_FIELD_VALUES
    grid->setFieldValue (FieldValue (currentValRe, currentValIm), coord, time_step_back);
#else /* COMPLEX_FIELD_VALUES */
    grid->setFieldValue (FieldValue (currentValRe), coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Load grid from file for specific layer for 2D.
 */
template<>
void
BMPLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> *grid,
                                           GridCoordinate2D startCoord,
                                           GridCoordinate2D endCoord,
                                           int time_step_back)
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
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);
  imageRe.ReadFromFile (imageReName.c_str());

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
  imageIm.ReadFromFile (imageImName.c_str());
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe = 0;
  FPValue maxPosRe = 0;
  FPValue maxNegRe = 0;
  std::ifstream fileMaxRe;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxIm = 0;
  FPValue maxPosIm = 0;
  FPValue maxNegIm = 0;
  std::ifstream fileMaxIm;
#endif /* COMPLEX_FIELD_VALUES */
  
  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosRe >> maxNegRe;
  fileMaxRe.close();
  maxRe = maxPosRe - maxNegRe;

#ifdef COMPLEX_FIELD_VALUES
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosIm >> maxNegIm;
  fileMaxIm.close();
  maxIm = maxPosIm - maxNegIm;
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set them.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());
      grid_coord coord = grid->calculateIndexFromPosition (pos);

      // Pixel coordinate.
      grid_coord px = pos.get1 ();
      grid_coord py = pos.get2 ();

      RGBApixel pixelRe = imageRe.GetPixel(px, py);
      FPValue currentValRe = BMPhelper.getValueFromPixel (pixelRe, maxNegRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
      RGBApixel pixelIm = imageIm.GetPixel(px, py);
      FPValue currentValIm = BMPhelper.getValueFromPixel (pixelIm, maxNegIm, maxIm);
#endif /* COMPLEX_FIELD_VALUES */

#ifdef COMPLEX_FIELD_VALUES
      grid->setFieldValue (FieldValue (currentValRe, currentValIm), coord, time_step_back);
#else /* COMPLEX_FIELD_VALUES */
      grid->setFieldValue (FieldValue (currentValRe), coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */
    }
  }
}

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Load grid from file for specific layer for 3D.
 */
template<>
void
BMPLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> *grid,
                                           GridCoordinate3D startCoord,
                                           GridCoordinate3D endCoord,
                                           int time_step_back)
{
  ASSERT_MESSAGE ("3D loader is not implemented as it is considered unneeded (see Docs/Input-Output.md).")
}

#endif /* MODE_DIM3 */

#include <iostream>

#include "BMPDumper.h"

/**
 * Save grid to file for specific layer for 1D.
 */
template<>
void
BMPDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> *grid, /**< grid to save */
                                          GridCoordinate1D startCoord, /**< start saving from this coordinate */
                                          GridCoordinate1D endCoord, /**< end saving at this coordinate */
                                          int time_step_back) /**< relative time step at which to save */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > startCoord.getZero () && endCoord <= grid->getSize ());

  grid_coord sx = grid->getSize ().get1 ();
  grid_coord sy = 1;

  std::string imageReName;
  std::string imageReNameTxt;
  std::string imageImName;
  std::string imageImNameTxt;
  std::string imageModName;
  std::string imageModNameTxt;
  setupNames (imageReName, imageReNameTxt, imageImName, imageImNameTxt, imageModName, imageModNameTxt, time_step_back, -1);

  // Create image for current values and max/min values.
  BMP imageRe;
  BMP imageIm;
  BMP imageMod;

  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);

  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (BMPHelper::bitDepth);
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe;
  FPValue minRe;
  FPValue maxIm;
  FPValue minIm;
  FPValue maxMod;
  FPValue minMod;

  getMaxValues (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                grid, startCoord, endCoord, time_step_back);

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());

    grid_coord px = pos.get1 ();
    grid_coord py = 0;

    setPixel (px, py, grid, pos, time_step_back, minRe, maxRe, minIm, maxIm, minMod, maxMod,
              imageRe, imageIm, imageMod);
  }

  imageRe.WriteToFile (imageReName.c_str ());
#ifdef COMPLEX_FIELD_VALUES
  imageIm.WriteToFile (imageImName.c_str ());
  imageMod.WriteToFile (imageModName.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

  writeTxtToFile (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                  imageReNameTxt, imageImNameTxt, imageModNameTxt);
} /* BMPDumper::writeToFile */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Save grid to file for specific layer for 2D.
 */
template<>
void
BMPDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> *grid, /**< grid to save */
                                          GridCoordinate2D startCoord, /**< start saving from this coordinate */
                                          GridCoordinate2D endCoord, /**< end saving at this coordinate */
                                          int time_step_back) /**< relative time step at which to save */
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

  // Create image for current values and max/min values.
  BMP imageRe;
  BMP imageIm;
  BMP imageMod;

  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);

  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (BMPHelper::bitDepth);
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe;
  FPValue minRe;
  FPValue maxIm;
  FPValue minIm;
  FPValue maxMod;
  FPValue minMod;

  // Go through all values and calculate max/min.
  getMaxValues (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                grid, startCoord, endCoord, time_step_back);

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());

      grid_coord px = pos.get1 ();
      grid_coord py = pos.get2 ();

      setPixel (px, py, grid, pos, time_step_back, minRe, maxRe, minIm, maxIm, minMod, maxMod,
                imageRe, imageIm, imageMod);
    }
  }

  imageRe.WriteToFile (imageReName.c_str ());
#ifdef COMPLEX_FIELD_VALUES
  imageIm.WriteToFile (imageImName.c_str ());
  imageMod.WriteToFile (imageModName.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

  writeTxtToFile (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                  imageReNameTxt, imageImNameTxt, imageModNameTxt);
} /* BMPDumper::writeToFile */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Save grid to file for specific layer for 3D.
 */
template<>
void
BMPDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> *grid, /**< grid to save */
                                          GridCoordinate3D startCoord, /**< start saving from this coordinate */
                                          GridCoordinate3D endCoord, /**< end saving at this coordinate */
                                          int time_step_back) /**< relative time step at which to save */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > startCoord.getZero () && endCoord <= grid->getSize ());

  grid_coord sx = grid->getSize ().get1 ();
  grid_coord sy = grid->getSize ().get2 ();
  grid_coord sz = grid->getSize ().get3 ();

  FPValue maxRe;
  FPValue minRe;
  FPValue maxIm;
  FPValue minIm;
  FPValue maxMod;
  FPValue minMod;

  // Go through all values and calculate max/min.
  getMaxValues (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                grid, startCoord, endCoord, time_step_back);

  grid_coord coordStart1, coordEnd1;
  grid_coord coordStart2, coordEnd2;
  grid_coord coordStart3, coordEnd3;

  grid_coord size1;
  grid_coord size2;
  grid_coord size3;

  if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::X)
  {
    coordStart1 = startCoord.get1 ();
    coordEnd1 = endCoord.get1 ();
    size1 = sx;

    coordStart2 = startCoord.get2 ();
    coordEnd2 = endCoord.get2 ();
    size2 = sy;

    coordStart3 = startCoord.get3 ();
    coordEnd3 = endCoord.get3 ();
    size3 = sz;
  }
  else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Y)
  {
    coordStart1 = startCoord.get2 ();
    coordEnd1 = endCoord.get2 ();
    size1 = sy;

    coordStart2 = startCoord.get1 ();
    coordEnd2 = endCoord.get1 ();
    size2 = sx;

    coordStart3 = startCoord.get3 ();
    coordEnd3 = endCoord.get3 ();
    size3 = sz;
  }
  else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Z)
  {
    coordStart1 = startCoord.get3 ();
    coordEnd1 = endCoord.get3 ();
    size1 = sz;

    coordStart2 = startCoord.get1 ();
    coordEnd2 = endCoord.get1 ();
    size2 = sx;

    coordStart3 = startCoord.get2 ();
    coordEnd3 = endCoord.get2 ();
    size3 = sy;
  }

  // Create image for current values and max/min values.
  for (grid_coord coord1 = coordStart1; coord1 < coordEnd1; ++coord1)
  {
    std::string imageReName;
    std::string imageReNameTxt;
    std::string imageImName;
    std::string imageImNameTxt;
    std::string imageModName;
    std::string imageModNameTxt;
    setupNames (imageReName, imageReNameTxt, imageImName, imageImNameTxt, imageModName, imageModNameTxt, time_step_back, coord1);

    BMP imageRe;
    BMP imageIm;
    BMP imageMod;

    imageRe.SetSize (size2, size3);
    imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
    imageIm.SetSize (size2, size3);
    imageIm.SetBitDepth (BMPHelper::bitDepth);

    imageMod.SetSize (size2, size3);
    imageMod.SetBitDepth (BMPHelper::bitDepth);
#endif /* COMPLEX_FIELD_VALUES */

    for (grid_coord coord2 = coordStart2; coord2 < coordEnd2; ++coord2)
    {
      for (grid_coord coord3 = coordStart3; coord3 < coordEnd3; ++coord3)
      {
        GridCoordinate3D pos;

        if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::X)
        {
          pos = GRID_COORDINATE_3D (coord1, coord2, coord3,
                                    grid->getSize ().getType1 (),
                                    grid->getSize ().getType2 (),
                                    grid->getSize ().getType3 ());
        }
        else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Y)
        {
          pos = GRID_COORDINATE_3D (coord2, coord1, coord3,
                                    grid->getSize ().getType1 (),
                                    grid->getSize ().getType2 (),
                                    grid->getSize ().getType3 ());
        }
        else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Z)
        {
          pos = GRID_COORDINATE_3D (coord2, coord3, coord1,
                                    grid->getSize ().getType1 (),
                                    grid->getSize ().getType2 (),
                                    grid->getSize ().getType3 ());
        }

        setPixel (coord2, coord3, grid, pos, time_step_back, minRe, maxRe, minIm, maxIm, minMod, maxMod,
                  imageRe, imageIm, imageMod);
      }
    }

    imageRe.WriteToFile (imageReName.c_str ());
#ifdef COMPLEX_FIELD_VALUES
    imageIm.WriteToFile (imageImName.c_str ());
    imageMod.WriteToFile (imageModName.c_str ());
#endif /* COMPLEX_FIELD_VALUES */
  }

  std::string imageReName;
  std::string imageReNameTxt;
  std::string imageImName;
  std::string imageImNameTxt;
  std::string imageModName;
  std::string imageModNameTxt;
  setupNames (imageReName, imageReNameTxt, imageImName, imageImNameTxt, imageModName, imageModNameTxt, time_step_back, -1);

  writeTxtToFile (maxRe, minRe, maxIm, minIm, maxMod, minMod,
                  imageReNameTxt, imageImNameTxt, imageModNameTxt);
} /* BMPDumper::writeToFile */

#endif /* MODE_DIM3 */

#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>

#include "BMPDumper.h"

/**
 * Save grid to file for specific layer for 1D.
 */
template<>
void
BMPDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> *grid,
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

  // Create image for current values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);

  BMP imageMod;
  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (BMPHelper::bitDepth);
#endif /* COMPLEX_FIELD_VALUES */

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosRe = grid->getFieldValue (startCoord, time_step_back)->real ();
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;

  FPValue maxPosIm = grid->getFieldValue (startCoord, time_step_back)->imag ();
  FPValue maxNegIm = maxPosIm;
  std::ofstream fileMaxIm;

  FPValue maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
  FPValue maxNegMod = maxPosMod;
  std::ofstream fileMaxMod;
#else /* COMPLEX_FIELD_VALUES */
  FPValue maxPosRe = *grid->getFieldValue (startCoord, time_step_back);
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;
#endif /* !COMPLEX_FIELD_VALUES */

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());
    grid_coord coord = grid->calculateIndexFromPosition (pos);

#ifdef COMPLEX_FIELD_VALUES
    FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
    FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
    FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
    FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

    if (valueRe > maxPosRe)
    {
      maxPosRe = valueRe;
    }
    if (valueRe < maxNegRe)
    {
      maxNegRe = valueRe;
    }

#ifdef COMPLEX_FIELD_VALUES
    if (valueIm > maxPosIm)
    {
      maxPosIm = valueIm;
    }
    if (valueIm < maxNegIm)
    {
      maxNegIm = valueIm;
    }

    if (valueMod > maxPosMod)
    {
      maxPosMod = valueMod;
    }
    if (valueMod < maxNegMod)
    {
      maxNegMod = valueMod;
    }
#endif /* COMPLEX_FIELD_VALUES */
  }

  // Set max (diff between max positive and max negative).
  const FPValue maxRe = maxPosRe - maxNegRe;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg " FP_MOD ", maxRe pos " FP_MOD ", maxRe " FP_MOD "\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;
  const FPValue maxMod = maxPosMod - maxNegMod;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg " FP_MOD ", maxIm pos " FP_MOD ", maxIm " FP_MOD "\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg " FP_MOD ", maxMod pos " FP_MOD ", maxMod " FP_MOD "\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());
    grid_coord coord = grid->calculateIndexFromPosition (pos);

    // Pixel coordinate.
    grid_coord px = pos.get1 ();
    grid_coord py = 0;

    // Get pixel for image.
#ifdef COMPLEX_FIELD_VALUES
    FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
    FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
    FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
    FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

    RGBApixel pixelRe = BMPhelper.getPixelFromValue (valueRe, maxNegRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
    RGBApixel pixelIm = BMPhelper.getPixelFromValue (valueIm, maxNegIm, maxIm);
    RGBApixel pixelMod = BMPhelper.getPixelFromValue (valueMod, maxNegMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

    // Set pixel for current image.
    imageRe.SetPixel(px, py, pixelRe);
#ifdef COMPLEX_FIELD_VALUES
    imageIm.SetPixel(px, py, pixelIm);
    imageMod.SetPixel(px, py, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
  }

  imageRe.WriteToFile (imageReName.c_str ());
  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe << std::setprecision(std::numeric_limits<double>::digits10) << maxPosRe << " " << maxNegRe;
  fileMaxRe.close();

#ifdef COMPLEX_FIELD_VALUES
  imageIm.WriteToFile (imageImName.c_str ());
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm << std::setprecision(std::numeric_limits<double>::digits10) << maxPosIm << " " << maxNegIm;
  fileMaxIm.close();

  imageMod.WriteToFile (imageModName.c_str ());
  fileMaxMod.open (imageModNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxMod.is_open());
  fileMaxMod << std::setprecision(std::numeric_limits<double>::digits10) << maxPosMod << " " << maxNegMod;
  fileMaxMod.close();
#endif /* COMPLEX_FIELD_VALUES */
}

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Save grid to file for specific layer for 2D.
 */
template<>
void
BMPDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> *grid,
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

  // Create image for current values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);

  BMP imageMod;
  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (BMPHelper::bitDepth);
#endif /* COMPLEX_FIELD_VALUES */

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosRe = grid->getFieldValue (startCoord, time_step_back)->real ();
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;

  FPValue maxPosIm = grid->getFieldValue (startCoord, time_step_back)->imag ();
  FPValue maxNegIm = maxPosIm;
  std::ofstream fileMaxIm;

  FPValue maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
  FPValue maxNegMod = maxPosMod;
  std::ofstream fileMaxMod;
#else /* COMPLEX_FIELD_VALUES */
  FPValue maxPosRe = *grid->getFieldValue (startCoord, time_step_back);
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;
#endif /* !COMPLEX_FIELD_VALUES */

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());
      grid_coord coord = grid->calculateIndexFromPosition (pos);

#ifdef COMPLEX_FIELD_VALUES
      FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
      FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
      FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
      FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

      if (valueRe > maxPosRe)
      {
        maxPosRe = valueRe;
      }
      if (valueRe < maxNegRe)
      {
        maxNegRe = valueRe;
      }

#ifdef COMPLEX_FIELD_VALUES
      if (valueIm > maxPosIm)
      {
        maxPosIm = valueIm;
      }
      if (valueIm < maxNegIm)
      {
        maxNegIm = valueIm;
      }

      if (valueMod > maxPosMod)
      {
        maxPosMod = valueMod;
      }
      if (valueMod < maxNegMod)
      {
        maxNegMod = valueMod;
      }
#endif /* COMPLEX_FIELD_VALUES */
    }
  }

  // Set max (diff between max positive and max negative).
  const FPValue maxRe = maxPosRe - maxNegRe;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg " FP_MOD ", maxRe pos " FP_MOD ", maxRe " FP_MOD "\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;
  const FPValue maxMod = maxPosMod - maxNegMod;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg " FP_MOD ", maxIm pos " FP_MOD ", maxIm " FP_MOD "\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg " FP_MOD ", maxMod pos " FP_MOD ", maxMod " FP_MOD "\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());
      grid_coord coord = grid->calculateIndexFromPosition (pos);

      // Pixel coordinate.
      grid_coord px = pos.get1 ();
      grid_coord py = pos.get2 ();

      // Get pixel for image.
#ifdef COMPLEX_FIELD_VALUES
      FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
      FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
      FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
      FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

      RGBApixel pixelRe = BMPhelper.getPixelFromValue (valueRe, maxNegRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
      RGBApixel pixelIm = BMPhelper.getPixelFromValue (valueIm, maxNegIm, maxIm);
      RGBApixel pixelMod = BMPhelper.getPixelFromValue (valueMod, maxNegMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

      // Set pixel for current image.
      imageRe.SetPixel(px, py, pixelRe);
#ifdef COMPLEX_FIELD_VALUES
      imageIm.SetPixel(px, py, pixelIm);
      imageMod.SetPixel(px, py, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
    }
  }

  imageRe.WriteToFile (imageReName.c_str ());
  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe << std::setprecision(std::numeric_limits<double>::digits10) << maxPosRe << " " << maxNegRe;
  fileMaxRe.close();

#ifdef COMPLEX_FIELD_VALUES
  imageIm.WriteToFile (imageImName.c_str ());
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm << std::setprecision(std::numeric_limits<double>::digits10) << maxPosIm << " " << maxNegIm;
  fileMaxIm.close();

  imageMod.WriteToFile (imageModName.c_str ());
  fileMaxMod.open (imageModNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxMod.is_open());
  fileMaxMod << std::setprecision(std::numeric_limits<double>::digits10) << maxPosMod << " " << maxNegMod;
  fileMaxMod.close();
#endif /* COMPLEX_FIELD_VALUES */
}

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Save grid to file for specific layer for 3D.
 */
template<>
void
BMPDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> *grid,
                                          GridCoordinate3D startCoord,
                                          GridCoordinate3D endCoord,
                                          int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && endCoord <= grid->getSize ());

  grid_coord sx = grid->getSize ().get1 ();
  grid_coord sy = grid->getSize ().get2 ();
  grid_coord sz = grid->getSize ().get3 ();

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosRe = grid->getFieldValue (startCoord, time_step_back)->real ();
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;

  FPValue maxPosIm = grid->getFieldValue (startCoord, time_step_back)->imag ();
  FPValue maxNegIm = maxPosIm;
  std::ofstream fileMaxIm;

  FPValue maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
  FPValue maxNegMod = maxPosMod;
  std::ofstream fileMaxMod;
#else /* COMPLEX_FIELD_VALUES */
  FPValue maxPosRe = *grid->getFieldValue (startCoord, time_step_back);
  FPValue maxNegRe = maxPosRe;
  std::ofstream fileMaxRe;
#endif /* !COMPLEX_FIELD_VALUES */

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      for (grid_coord k = startCoord.get3 (); k < endCoord.get3 (); ++k)
      {
        GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k,
                                                   grid->getSize ().getType1 (),
                                                   grid->getSize ().getType2 (),
                                                   grid->getSize ().getType3 ());
        grid_coord coord = grid->calculateIndexFromPosition (pos);

#ifdef COMPLEX_FIELD_VALUES
        FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
        FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
        FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

        if (valueRe > maxPosRe)
        {
          maxPosRe = valueRe;
        }
        if (valueRe < maxNegRe)
        {
          maxNegRe = valueRe;
        }

#ifdef COMPLEX_FIELD_VALUES
        if (valueIm > maxPosIm)
        {
          maxPosIm = valueIm;
        }
        if (valueIm < maxNegIm)
        {
          maxNegIm = valueIm;
        }

        if (valueMod > maxPosMod)
        {
          maxPosMod = valueMod;
        }
        if (valueMod < maxNegMod)
        {
          maxNegMod = valueMod;
        }
#endif /* COMPLEX_FIELD_VALUES */
      }
    }
  }

  // Set max (diff between max positive and max negative).
  const FPValue maxRe = maxPosRe - maxNegRe;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg " FP_MOD ", maxRe pos " FP_MOD ", maxRe " FP_MOD "\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;
  const FPValue maxMod = maxPosMod - maxNegMod;
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg " FP_MOD ", maxIm pos " FP_MOD ", maxIm " FP_MOD "\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg " FP_MOD ", maxMod pos " FP_MOD ", maxMod " FP_MOD "\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

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
    imageRe.SetSize (size2, size3);
    imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
    BMP imageIm;
    imageIm.SetSize (size2, size3);
    imageIm.SetBitDepth (BMPHelper::bitDepth);

    BMP imageMod;
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

        grid_coord coord = grid->calculateIndexFromPosition (pos);

        // Get pixel for image.
#ifdef COMPLEX_FIELD_VALUES
        FPValue valueRe = grid->getFieldValue (coord, time_step_back)->real ();
        FPValue valueIm = grid->getFieldValue (coord, time_step_back)->imag ();
        FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        FPValue valueRe = *grid->getFieldValue (coord, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

        RGBApixel pixelRe = BMPhelper.getPixelFromValue (valueRe, maxNegRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
        RGBApixel pixelIm = BMPhelper.getPixelFromValue (valueIm, maxNegIm, maxIm);
        RGBApixel pixelMod = BMPhelper.getPixelFromValue (valueMod, maxNegMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

        // Set pixel for current image.
        imageRe.SetPixel(coord2, coord3, pixelRe);
#ifdef COMPLEX_FIELD_VALUES
        imageIm.SetPixel(coord2, coord3, pixelIm);
        imageMod.SetPixel(coord2, coord3, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
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

  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe << std::setprecision(std::numeric_limits<double>::digits10) << maxPosRe << " " << maxNegRe;
  fileMaxRe.close();

#ifdef COMPLEX_FIELD_VALUES
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm << std::setprecision(std::numeric_limits<double>::digits10) << maxPosIm << " " << maxNegIm;
  fileMaxIm.close();

  fileMaxMod.open (imageModNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxMod.is_open());
  fileMaxMod << std::setprecision(std::numeric_limits<double>::digits10) << maxPosMod << " " << maxNegMod;
  fileMaxMod.close();
#endif /* COMPLEX_FIELD_VALUES */
}

#endif /* MODE_DIM3 */

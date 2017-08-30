#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>

#include "BMPLoader.h"

/**
 * Virtual method for grid loading for 1D
 */
template<>
void
BMPLoader<GridCoordinate1D>::loadGrid (Grid<GridCoordinate1D> *grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate1D& size = grid->getSize ();
  grid_coord sx = size.getX ();

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Loading 1D from BMP image. Size: %ux1x1\n", sx);
#endif /* PRINT_MESSAGE */

  loadFromFile (grid);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Loaded.\n");
}

/**
 * Virtual method for grid loading for 2D
 */
template<>
void
BMPLoader<GridCoordinate2D>::loadGrid (Grid<GridCoordinate2D> *grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate2D& size = grid->getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Loading 2D from BMP image. Size: %ux%ux1\n", sx, sy)
#endif /* PRINT_MESSAGE */

  loadFromFile (grid);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Loaded.\n");
}

/**
 * Virtual method for grid loading for 3D
 */
template<>
void
BMPLoader<GridCoordinate3D>::loadGrid (Grid<GridCoordinate3D> *grid) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

/**
 * Load grid from file for specific layer for 1D.
 */
template<>
void
BMPLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid, GridFileType load_type) const
{
  const GridCoordinate1D& size = grid->getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = 1;

  // Create image for values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
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

  switch (load_type)
  {
    case CURRENT:
    {
      std::string cur_txt = cur + std::string ("-Re") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (cur_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      cur_txt = cur + std::string ("-Im") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (cur_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_txt =  prev + std::string ("-Re") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (prev_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      prev_txt = prev + std::string ("-Im") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (prev_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_txt = prevPrev + std::string ("-Mod") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (prevPrev_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      prevPrev_txt = prevPrev + std::string ("-Mod") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (prevPrev_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (fileMaxRe.is_open());
  fileMaxRe >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosRe >> maxNegRe;
  fileMaxRe.close();

  maxRe = maxPosRe - maxNegRe;
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (fileMaxIm.is_open());
  fileMaxIm >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosIm >> maxNegIm;
  fileMaxIm.close();

  maxIm = maxPosIm - maxNegIm;
#endif

  switch (load_type)
  {
    case CURRENT:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  // Go through all values and set them.
  grid_iter end = grid->getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = grid->getFieldPointValue (iter);
    ASSERT (current);

    // Calculate its position from index in array.
    GridCoordinate1D coord = grid->calculatePositionFromIndex (iter);

    // Pixel coordinate.
    grid_iter px = coord.getX ();
    grid_iter py = 0;

    RGBApixel pixelRe = imageRe.GetPixel(px, py);

    // Get pixel for image.
    FPValue currentValRe = BMPhelper.getValueFromPixel (pixelRe, maxNegRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
    RGBApixel pixelIm = imageIm.GetPixel(px, py);

    FPValue currentValIm = BMPhelper.getValueFromPixel (pixelIm, maxNegIm, maxIm);
#endif /* COMPLEX_FIELD_VALUES */

    switch (load_type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setCurValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setCurValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setPrevValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setPrevValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setPrevPrevValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setPrevPrevValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }
  }
}

/**
 * Load grid from file for specific layer for 2D.
 */
template<>
void
BMPLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> *grid, GridFileType load_type) const
{
  const GridCoordinate2D& size = grid->getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  // Create image for values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (BMPHelper::bitDepth);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (BMPHelper::bitDepth);
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

  switch (load_type)
  {
    case CURRENT:
    {
      std::string cur_txt = cur + std::string ("-Re") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (cur_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      cur_txt = cur + std::string ("-Im") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (cur_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_txt =  prev + std::string ("-Re") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (prev_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      prev_txt = prev + std::string ("-Im") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (prev_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_txt = prevPrev + std::string ("-Mod") + std::string (".bmp") + std::string (".txt");
      fileMaxRe.open (prevPrev_txt.c_str (), std::ios::in);

#ifdef COMPLEX_FIELD_VALUES
      prevPrev_txt = prevPrev + std::string ("-Mod") + std::string (".bmp") + std::string (".txt");
      fileMaxIm.open (prevPrev_txt.c_str (), std::ios::in);
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (fileMaxRe.is_open());
  fileMaxRe >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosRe >> maxNegRe;
  fileMaxRe.close();

  maxRe = maxPosRe - maxNegRe;
  #ifdef COMPLEX_FIELD_VALUES
  ASSERT (fileMaxIm.is_open());
  fileMaxIm >> std::setprecision(std::numeric_limits<double>::digits10) >> maxPosIm >> maxNegIm;
  fileMaxIm.close();

  maxIm = maxPosIm - maxNegIm;
  #endif

  switch (load_type)
  {
    case CURRENT:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  // Go through all values and set them.
  grid_iter end = grid->getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = grid->getFieldPointValue (iter);
    ASSERT (current);

    // Calculate its position from index in array.
    GridCoordinate2D coord = grid->calculatePositionFromIndex (iter);

    // Pixel coordinate.
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();

    RGBApixel pixelRe = imageRe.GetPixel(px, py);

    // Get pixel for image.
    FPValue currentValRe = BMPhelper.getValueFromPixel (pixelRe, maxNegRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
    RGBApixel pixelIm = imageIm.GetPixel(px, py);

    FPValue currentValIm = BMPhelper.getValueFromPixel (pixelIm, maxNegIm, maxIm);
#endif /* COMPLEX_FIELD_VALUES */

    switch (load_type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setCurValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setCurValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setPrevValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setPrevValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        current->setPrevPrevValue (FieldValue (currentValRe, currentValIm));
#else /* COMPLEX_FIELD_VALUES */
        current->setPrevPrevValue (currentValRe);
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }
  }
}

/**
 * Load grid from file for specific layer for 3D.
 */
template<>
void
BMPLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> *grid, GridFileType load_type) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

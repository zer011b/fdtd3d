#include <iostream>

#include "BMPLoader.h"

/**
 * Virtual method for grid loading for 1D
 */
template<>
void
BMPLoader<GridCoordinate1D>::loadGrid (Grid<GridCoordinate1D> &grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate1D& size = grid.getSize ();
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
BMPLoader<GridCoordinate2D>::loadGrid (Grid<GridCoordinate2D> &grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate2D& size = grid.getSize ();
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
BMPLoader<GridCoordinate3D>::loadGrid (Grid<GridCoordinate3D> &grid) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

/**
 * Load grid from file for specific layer for 1D.
 */
template<>
void
BMPLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> &grid, GridFileType load_type) const
{
  const GridCoordinate1D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = 1;

  // Create image for values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe = 0;
  FPValue maxNegRe = 0;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxIm = 0;
  FPValue maxNegIm = 0;
#endif /* COMPLEX_FIELD_VALUES */

  switch (load_type)
  {
    case CURRENT:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getCurValue ().real () - maxValueNeg.getCurValue ().real ();
      maxNegRe = maxValueNeg.getCurValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getCurValue ().imag () - maxValueNeg.getCurValue ().imag ();
      maxNegIm = maxValueNeg.getCurValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getCurValue () - maxValueNeg.getCurValue ();
      maxNegRe = maxValueNeg.getCurValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getPrevValue ().real () - maxValueNeg.getPrevValue ().real ();
      maxNegRe = maxValueNeg.getPrevValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getPrevValue ().imag () - maxValueNeg.getPrevValue ().imag ();
      maxNegIm = maxValueNeg.getPrevValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getPrevValue () - maxValueNeg.getPrevValue ();
      maxNegRe = maxValueNeg.getPrevValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getPrevPrevValue ().real () - maxValueNeg.getPrevPrevValue ().real ();
      maxNegRe = maxValueNeg.getPrevPrevValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getPrevPrevValue ().imag () - maxValueNeg.getPrevPrevValue ().imag ();
      maxNegIm = maxValueNeg.getPrevPrevValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getPrevPrevValue () - maxValueNeg.getPrevPrevValue ();
      maxNegRe = maxValueNeg.getPrevPrevValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
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
  grid_iter end = grid.getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = grid.getFieldPointValue (iter);
    ASSERT (current);

    // Calculate its position from index in array.
    GridCoordinate1D coord = grid.calculatePositionFromIndex (iter);

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
BMPLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> &grid, GridFileType load_type) const
{
  const GridCoordinate2D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  // Create image for values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

  FPValue maxRe = 0;
  FPValue maxNegRe = 0;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxIm = 0;
  FPValue maxNegIm = 0;
#endif /* COMPLEX_FIELD_VALUES */

  switch (load_type)
  {
    case CURRENT:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getCurValue ().real () - maxValueNeg.getCurValue ().real ();
      maxNegRe = maxValueNeg.getCurValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getCurValue ().imag () - maxValueNeg.getCurValue ().imag ();
      maxNegIm = maxValueNeg.getCurValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getCurValue () - maxValueNeg.getCurValue ();
      maxNegRe = maxValueNeg.getCurValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getPrevValue ().real () - maxValueNeg.getPrevValue ().real ();
      maxNegRe = maxValueNeg.getPrevValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getPrevValue ().imag () - maxValueNeg.getPrevValue ().imag ();
      maxNegIm = maxValueNeg.getPrevValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getPrevValue () - maxValueNeg.getPrevValue ();
      maxNegRe = maxValueNeg.getPrevValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxRe = maxValuePos.getPrevPrevValue ().real () - maxValueNeg.getPrevPrevValue ().real ();
      maxNegRe = maxValueNeg.getPrevPrevValue ().real ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());

      maxIm = maxValuePos.getPrevPrevValue ().imag () - maxValueNeg.getPrevPrevValue ().imag ();
      maxNegIm = maxValueNeg.getPrevPrevValue ().imag ();

      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.ReadFromFile (cur_bmp_im.c_str());
#else /* COMPLEX_FIELD_VALUES */
      maxRe = maxValuePos.getPrevPrevValue () - maxValueNeg.getPrevPrevValue ();
      maxNegRe = maxValueNeg.getPrevPrevValue ();

      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.ReadFromFile (cur_bmp_re.c_str());
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
  grid_iter end = grid.getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue* current = grid.getFieldPointValue (iter);
    ASSERT (current);

    // Calculate its position from index in array.
    GridCoordinate2D coord = grid.calculatePositionFromIndex (iter);

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
BMPLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> &grid, GridFileType load_type) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

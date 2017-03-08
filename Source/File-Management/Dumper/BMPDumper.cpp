#include <iostream>

#include "BMPDumper.h"

/**
 * Virtual method for grid saving for 1D
 */
template<>
void
BMPDumper<GridCoordinate1D>::dumpGrid (Grid<GridCoordinate1D> &grid,
                                       GridCoordinate1D startCoord,
                                       GridCoordinate1D endCoord) const
{
#if PRINT_MESSAGE
  const GridCoordinate1D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  std::cout << "Saving 1D to BMP image. Size: " << sx << "x1. " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Virtual method for grid saving for 2D
 */
template<>
void
BMPDumper<GridCoordinate2D>::dumpGrid (Grid<GridCoordinate2D> &grid,
                                       GridCoordinate2D startCoord,
                                       GridCoordinate2D endCoord) const
{
#if PRINT_MESSAGE
  const GridCoordinate2D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Saving 2D to BMP image. Size: " << sx << "x" << sy << ". " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Virtual method for grid saving for 3D
 */
template<>
void
BMPDumper<GridCoordinate3D>::dumpGrid (Grid<GridCoordinate3D> &grid,
                                       GridCoordinate3D startCoord,
                                       GridCoordinate3D endCoord) const
{
//  ASSERT_MESSAGE ("3D Dumper is not implemented.")
#if PRINT_MESSAGE
  const GridCoordinate3D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();
  grid_coord sz = size.getZ ();

  std::cout << "Saving 3D to BMP image. Size: " << sx << "x" << sy << "x" << sz << ". " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Save grid to file for specific layer for 1D.
 */
template<>
void
BMPDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> &grid, GridFileType dump_type, GridCoordinate1D startCoord, GridCoordinate1D endCoord) const
{
  const GridCoordinate1D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = 1;

  // Create image for current values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (24);

  BMP imageMod;
  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

  const FieldPointValue* value0 = grid.getFieldPointValue (startCoord);
  ASSERT (value0);

  FPValue maxPosRe = 0;
  FPValue maxNegRe = 0;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosIm = 0;
  FPValue maxNegIm = 0;

  FPValue maxPosMod = 0;
  FPValue maxNegMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

  switch (dump_type)
  {
    case CURRENT:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getCurValue ().real ();
      maxNegIm = maxPosIm = value0->getCurValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevPrevValue ();
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

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.getX (); i < endCoord.getX (); ++i)
  {
    const FieldPointValue* current = grid.getFieldPointValue (GridCoordinate1D (i));

    ASSERT (current);

    FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
    FPValue valueIm = 0;

    FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

    switch (dump_type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getCurValue ().real ();
        valueIm = current->getCurValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getPrevValue ().real ();
        valueIm = current->getPrevValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getPrevPrevValue ().real ();
        valueIm = current->getPrevPrevValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getPrevPrevValue ();
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

  printf ("MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  printf ("MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  printf ("MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.getX (); i < endCoord.getX (); ++i)
  {
    // Get current point value.
    GridCoordinate1D coord (i);
    const FieldPointValue* current = grid.getFieldPointValue (coord);
    ASSERT (current);

    // Pixel coordinate.
    grid_iter px = coord.getX ();
    grid_iter py = 0;

    // Get pixel for image.
    FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
    FPValue valueIm = 0;

    FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

    switch (dump_type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getCurValue ().real ();
        valueIm = current->getCurValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getPrevValue ().real ();
        valueIm = current->getPrevValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        valueRe = current->getPrevPrevValue ().real ();
        valueIm = current->getPrevPrevValue ().imag ();

        valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
        valueRe = current->getPrevPrevValue ();
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

  // Write image to file.
  switch (dump_type)
  {
    case CURRENT:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (cur_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (cur_bmp_im.c_str ());

      std::string cur_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (cur_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (prev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string prev_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (prev_bmp_im.c_str ());

      std::string prev_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (prev_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (prevPrev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string prevPrev_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (prevPrev_bmp_im.c_str ());

      std::string prevPrev_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (prevPrev_bmp_mod.c_str ());
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
}

/**
 * Save grid to file for specific layer for 2D.
 */
template<>
void
BMPDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> &grid, GridFileType dump_type, GridCoordinate2D startCoord, GridCoordinate2D endCoord) const
{
  const GridCoordinate2D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();;

  // Create image for current values and max/min values.
  BMP imageRe;
  imageRe.SetSize (sx, sy);
  imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
  BMP imageIm;
  imageIm.SetSize (sx, sy);
  imageIm.SetBitDepth (24);

  BMP imageMod;
  imageMod.SetSize (sx, sy);
  imageMod.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

  const FieldPointValue* value0 = grid.getFieldPointValue (startCoord);
  ASSERT (value0);

  FPValue maxPosRe = 0;
  FPValue maxNegRe = 0;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosIm = 0;
  FPValue maxNegIm = 0;

  FPValue maxPosMod = 0;
  FPValue maxNegMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

  switch (dump_type)
  {
    case CURRENT:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getCurValue ().real ();
      maxNegIm = maxPosIm = value0->getCurValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevPrevValue ();
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

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.getX (); i < endCoord.getX (); ++i)
  {
    for (grid_coord j = startCoord.getY (); j < endCoord.getY (); ++j)
    {
      const FieldPointValue* current = grid.getFieldPointValue (GridCoordinate2D (i, j));

      ASSERT (current);

      FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
      FPValue valueIm = 0;

      FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

      switch (dump_type)
      {
        case CURRENT:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getCurValue ().real ();
          valueIm = current->getCurValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          break;
        }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        case PREVIOUS:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getPrevValue ().real ();
          valueIm = current->getPrevValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          break;
        }
#if defined (TWO_TIME_STEPS)
        case PREVIOUS2:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getPrevPrevValue ().real ();
          valueIm = current->getPrevPrevValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getPrevPrevValue ();
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

  printf ("MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  printf ("MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  printf ("MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Go through all values and set pixels.
  for (grid_coord i = startCoord.getX (); i < endCoord.getX (); ++i)
  {
    for (grid_coord j = startCoord.getY (); j < endCoord.getY (); ++j)
    {
      GridCoordinate2D coord (i, j);

      // Get current point value.
      const FieldPointValue* current = grid.getFieldPointValue (coord);
      ASSERT (current);

      // Pixel coordinate.
      grid_iter px = coord.getX ();
      grid_iter py = coord.getY ();;

      // Get pixel for image.
      FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
      FPValue valueIm = 0;

      FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

      switch (dump_type)
      {
        case CURRENT:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getCurValue ().real ();
          valueIm = current->getCurValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          break;
        }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        case PREVIOUS:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getPrevValue ().real ();
          valueIm = current->getPrevValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

          break;
        }
#if defined (TWO_TIME_STEPS)
        case PREVIOUS2:
        {
#ifdef COMPLEX_FIELD_VALUES
          valueRe = current->getPrevPrevValue ().real ();
          valueIm = current->getPrevPrevValue ().imag ();

          valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
          valueRe = current->getPrevPrevValue ();
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

  // Write image to file.
  switch (dump_type)
  {
    case CURRENT:
    {
      std::string cur_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (cur_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string cur_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (cur_bmp_im.c_str ());

      std::string cur_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (cur_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (prev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string prev_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (prev_bmp_im.c_str ());

      std::string prev_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (prev_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_bmp_re = cur + std::string ("-Re") + std::string (".bmp");
      imageRe.WriteToFile (prevPrev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
      std::string prevPrev_bmp_im = cur + std::string ("-Im") + std::string (".bmp");
      imageIm.WriteToFile (prevPrev_bmp_im.c_str ());

      std::string prevPrev_bmp_mod = cur + std::string ("-Mod") + std::string (".bmp");
      imageMod.WriteToFile (prevPrev_bmp_mod.c_str ());
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
}

/**
 * Save grid to file for specific layer for 3D.
 */
template<>
void
BMPDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> &grid, GridFileType dump_type, GridCoordinate3D startCoord, GridCoordinate3D endCoord) const
{
//  ASSERT_MESSAGE ("3D Dumper is not implemented.")

  const GridCoordinate3D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();
  grid_coord sz = size.getZ ();

  const FieldPointValue* value0 = grid.getFieldPointValue (startCoord);
  ASSERT (value0);

  FPValue maxPosRe = 0;
  FPValue maxNegRe = 0;

#ifdef COMPLEX_FIELD_VALUES
  FPValue maxPosIm = 0;
  FPValue maxNegIm = 0;

  FPValue maxPosMod = 0;
  FPValue maxNegMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

  switch (dump_type)
  {
    case CURRENT:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getCurValue ().real ();
      maxNegIm = maxPosIm = value0->getCurValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef COMPLEX_FIELD_VALUES
      maxNegRe = maxPosRe = value0->getPrevPrevValue ().real ();
      maxNegIm = maxPosIm = value0->getPrevPrevValue ().imag ();

      maxNegMod = maxPosMod = sqrt (maxNegRe * maxNegRe + maxNegIm * maxNegIm);
#else /* COMPLEX_FIELD_VALUES */
      maxNegRe = maxPosRe = value0->getPrevPrevValue ();
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

  // Go through all values and calculate max/min.
  for (grid_coord i = startCoord.getX (); i < endCoord.getX (); ++i)
  {
    for (grid_coord j = startCoord.getY (); j < endCoord.getY (); ++j)
    {
      for (grid_coord k = startCoord.getZ (); k < endCoord.getZ (); ++k)
      {
        const FieldPointValue* current = grid.getFieldPointValue (GridCoordinate3D (i, j, k));

        ASSERT (current);

        FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
        FPValue valueIm = 0;

        FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

        switch (dump_type)
        {
          case CURRENT:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getCurValue ().real ();
            valueIm = current->getCurValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

            break;
          }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          case PREVIOUS:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getPrevValue ().real ();
            valueIm = current->getPrevValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

            break;
          }
#if defined (TWO_TIME_STEPS)
          case PREVIOUS2:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getPrevPrevValue ().real ();
            valueIm = current->getPrevPrevValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getPrevPrevValue ();
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

  printf ("MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  printf ("MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  printf ("MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Create image for current values and max/min values.
  for (grid_coord k = startCoord.getZ (); k < endCoord.getZ (); ++k)
  {
    BMP imageRe;
    imageRe.SetSize (sx, sy);
    imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
    BMP imageIm;
    imageIm.SetSize (sx, sy);
    imageIm.SetBitDepth (24);

    BMP imageMod;
    imageMod.SetSize (sx, sy);
    imageMod.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

    for (grid_iter i = startCoord.getX (); i < endCoord.getX (); ++i)
    {
      for (grid_iter j = startCoord.getY (); j < endCoord.getY (); ++j)
      {
        GridCoordinate3D pos (i, j, k);

        // Get current point value.
        const FieldPointValue* current = grid.getFieldPointValue (pos);
        ASSERT (current);

        // Get pixel for image.
        FPValue valueRe = 0;

#ifdef COMPLEX_FIELD_VALUES
        FPValue valueIm = 0;

        FPValue valueMod = 0;
#endif /* COMPLEX_FIELD_VALUES */

        switch (dump_type)
        {
          case CURRENT:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getCurValue ().real ();
            valueIm = current->getCurValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */

            break;
          }
  #if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          case PREVIOUS:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getPrevValue ().real ();
            valueIm = current->getPrevValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getPrevValue ();
#endif /* !COMPLEX_FIELD_VALUES */

            break;
          }
  #if defined (TWO_TIME_STEPS)
          case PREVIOUS2:
          {
#ifdef COMPLEX_FIELD_VALUES
            valueRe = current->getPrevPrevValue ().real ();
            valueIm = current->getPrevPrevValue ().imag ();

            valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
            valueRe = current->getPrevPrevValue ();
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

        RGBApixel pixelRe = BMPhelper.getPixelFromValue (valueRe, maxNegRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
        RGBApixel pixelIm = BMPhelper.getPixelFromValue (valueIm, maxNegIm, maxIm);

        RGBApixel pixelMod = BMPhelper.getPixelFromValue (valueMod, maxNegMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

        // Set pixel for current image.
        imageRe.SetPixel(i, j, pixelRe);

#ifdef COMPLEX_FIELD_VALUES
        imageIm.SetPixel(i, j, pixelIm);

        imageMod.SetPixel(i, j, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
      }
    }

    // Write image to file.
    switch (dump_type)
    {
      case CURRENT:
      {
        std::string cur_bmp_re = cur + int64_to_string (k) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (cur_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string cur_bmp_im = cur + int64_to_string (k) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (cur_bmp_im.c_str ());

        std::string cur_bmp_mod = cur + int64_to_string (k) + std::string ("-Mod") + std::string (".bmp");
        imageMod.WriteToFile (cur_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        std::string prev_bmp_re = cur + int64_to_string (k) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (prev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string prev_bmp_im = cur + int64_to_string (k) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (prev_bmp_im.c_str ());

        std::string prev_bmp_mod = cur + int64_to_string (k) + std::string ("-Mod") + std::string (".bmp");
        imageMod.WriteToFile (prev_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        std::string prevPrev_bmp_re = cur + int64_to_string (k) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (prevPrev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string prevPrev_bmp_im = cur + int64_to_string (k) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (prevPrev_bmp_im.c_str ());

        std::string prevPrev_bmp_mod = cur + int64_to_string (k) + std::string ("-Mod") + std::string (".bmp");
        imageMod.WriteToFile (prevPrev_bmp_mod.c_str ());
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
  }
}

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
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saving 1D Grid <%s> to BMP image. Size: %ux1x1\n", grid.getName ().c_str (), sx);
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saved,\n");
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saving 2D Grid <%s> to BMP image. Size: %ux%ux1\n", grid.getName ().c_str (), sx, sy);
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saved,\n");
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saving 2D Grid <%s> to BMP image. Size: %ux%ux%u\n", grid.getName ().c_str (), sx, sy, sz);
#endif /* PRINT_MESSAGE */

  writeToFile (grid, startCoord, endCoord);

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saved,\n");
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
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

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxRe neg %f, maxRe pos %f, maxRe %f\n", maxNegRe, maxPosRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  const FPValue maxIm = maxPosIm - maxNegIm;

  const FPValue maxMod = maxPosMod - maxNegMod;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxIm neg %f, maxIm pos %f, maxIm %f\n", maxNegIm, maxPosIm, maxIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "MaxMod neg %f, maxMod pos %f, maxMod %f\n", maxNegMod, maxPosMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  grid_coord coordStart1, coordEnd1;
  grid_coord coordStart2, coordEnd2;
  grid_coord coordStart3, coordEnd3;

  grid_coord size1;
  grid_coord size2;
  grid_coord size3;

  if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::X)
  {
    coordStart1 = startCoord.getX ();
    coordEnd1 = endCoord.getX ();
    size1 = sx;

    coordStart2 = startCoord.getY ();
    coordEnd2 = endCoord.getY ();
    size2 = sy;

    coordStart3 = startCoord.getZ ();
    coordEnd3 = endCoord.getZ ();
    size3 = sz;
  }
  else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Y)
  {
    coordStart1 = startCoord.getY ();
    coordEnd1 = endCoord.getY ();
    size1 = sy;

    coordStart2 = startCoord.getX ();
    coordEnd2 = endCoord.getX ();
    size2 = sx;

    coordStart3 = startCoord.getZ ();
    coordEnd3 = endCoord.getZ ();
    size3 = sz;
  }
  else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Z)
  {
    coordStart1 = startCoord.getZ ();
    coordEnd1 = endCoord.getZ ();
    size1 = sz;

    coordStart2 = startCoord.getX ();
    coordEnd2 = endCoord.getX ();
    size2 = sx;

    coordStart3 = startCoord.getY ();
    coordEnd3 = endCoord.getY ();
    size3 = sy;
  }

  // Create image for current values and max/min values.
  for (grid_coord coord1 = coordStart1; coord1 < coordEnd1; ++coord1)
  {
    BMP imageRe;
    imageRe.SetSize (size2, size3);
    imageRe.SetBitDepth (24);

#ifdef COMPLEX_FIELD_VALUES
    BMP imageIm;
    imageIm.SetSize (size2, size3);
    imageIm.SetBitDepth (24);

    BMP imageMod;
    imageMod.SetSize (size2, size3);
    imageMod.SetBitDepth (24);
#endif /* COMPLEX_FIELD_VALUES */

    for (grid_iter coord2 = coordStart2; coord2 < coordEnd2; ++coord2)
    {
      for (grid_iter coord3 = coordStart3; coord3 < coordEnd3; ++coord3)
      {
        GridCoordinate3D pos;

        if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::X)
        {
          pos = GridCoordinate3D (coord1, coord2, coord3);
        }
        else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Y)
        {
          pos = GridCoordinate3D (coord2, coord1, coord3);
        }
        else if (BMPhelper.getOrthogonalAxis () == OrthogonalAxis::Z)
        {
          pos = GridCoordinate3D (coord2, coord3, coord1);
        }

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
        imageRe.SetPixel(coord2, coord3, pixelRe);

#ifdef COMPLEX_FIELD_VALUES
        imageIm.SetPixel(coord2, coord3, pixelIm);

        imageMod.SetPixel(coord2, coord3, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
      }
    }

    // Write image to file.
    switch (dump_type)
    {
      case CURRENT:
      {
        std::string cur_bmp_re = cur + int64_to_string (coord1) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (cur_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string cur_bmp_im = cur + int64_to_string (coord1) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (cur_bmp_im.c_str ());

        std::string cur_bmp_mod = cur + int64_to_string (coord1) + std::string ("-Mod") + std::string (".bmp");
        imageMod.WriteToFile (cur_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        std::string prev_bmp_re = cur + int64_to_string (coord1) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (prev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string prev_bmp_im = cur + int64_to_string (coord1) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (prev_bmp_im.c_str ());

        std::string prev_bmp_mod = cur + int64_to_string (coord1) + std::string ("-Mod") + std::string (".bmp");
        imageMod.WriteToFile (prev_bmp_mod.c_str ());
#endif /* COMPLEX_FIELD_VALUES */

        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        std::string prevPrev_bmp_re = cur + int64_to_string (coord1) + std::string ("-Re") + std::string (".bmp");
        imageRe.WriteToFile (prevPrev_bmp_re.c_str ());

#ifdef COMPLEX_FIELD_VALUES
        std::string prevPrev_bmp_im = cur + int64_to_string (coord1) + std::string ("-Im") + std::string (".bmp");
        imageIm.WriteToFile (prevPrev_bmp_im.c_str ());

        std::string prevPrev_bmp_mod = cur + int64_to_string (coord1) + std::string ("-Mod") + std::string (".bmp");
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

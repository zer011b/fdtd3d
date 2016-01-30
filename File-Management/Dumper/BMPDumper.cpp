#include <iostream>

#include "BMPDumper.h"

void
BMPDumper::dumpGrid (Grid& grid) const
{
#if defined (GRID_1D)
  dump1D (grid);
#else /* GRID_1D */
#if defined (GRID_2D)
  dump2D (grid);
#else /* GRID_2D */
#if defined (GRID_3D)
  dump3D (grid);
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */
}

/*
 * Return pixel with colors according to values.
 * Blue-Green-Red scheme.
 */
RGBApixel
BMPDumper::getPixelFromValue (const FieldValue& val, const FieldValue& maxNeg,
                              const FieldValue& max) const
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FieldValue value = val - maxNeg;
  FieldValue max_2 = max / 2.0;
  if (value > max_2)
  {
    value -= max_2;
    FieldValue tmp = value / max_2;
    pixel.Red = tmp * 255;
    pixel.Green = (1.0 - tmp) * 255;
    pixel.Blue = 0.0;
  }
  else
  {
    FieldValue tmp = 0;
    if (max == 0)
    {
      tmp = 0.0;
    }
    else
    {
      tmp = value / max_2;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * 255;
    pixel.Blue = (1.0 - tmp) * 255;
  }

  return pixel;
}

void
BMPDumper::writeToFile (Grid& grid, const grid_iter& sx, const grid_iter& sy, GridFileType dump_type) const
{
  // Create image for current values and max/min values.
  BMP image;
  image.SetSize (sx, sy);
  image.SetBitDepth (24);

  const FieldPointValue* value0 = grid.getFieldPointValue (0);
  ASSERT (value0);

  FieldValue maxPos = 0;
  FieldValue maxNeg = 0;

  switch (dump_type)
  {
    case CURRENT:
    {
      maxNeg = maxPos = value0->getCurValue ();
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      maxNeg = maxPos = value0->getPrevValue ();
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      maxNeg = maxPos = value0->getPrevPrevValue ();
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
  for (grid_iter i = 0; i < grid.getSize ().calculateTotalCoord (); ++i)
  {
    const FieldPointValue* current = grid.getFieldPointValue (i);

    ASSERT (current);

    FieldValue value = 0;

    switch (dump_type)
    {
      case CURRENT:
      {
        value = current->getCurValue ();
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        value = current->getPrevValue ();
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        value = current->getPrevPrevValue ();
        break;
      }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }

    if (value > maxPos)
    {
      maxPos = value;
    }
    if (value < maxNeg)
    {
      maxNeg = value;
    }
  }

  // Set max (diff between max positive and max negative).
  const FieldValue max = maxPos - maxNeg;

  // Go through all values and set pixels.
  grid_iter end = grid.getSize().calculateTotalCoord ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    const FieldPointValue* current = grid.getFieldPointValue (iter);
    ASSERT (current);

    // Calculate its position from index in array.
    GridCoordinate coord = grid.calculatePositionFromIndex (iter);

    // Pixel coordinate.
#if defined (GRID_1D)
    grid_iter px = coord.getX ();
    grid_iter py = 0;
#else /* GRID_1D */
#if defined (GRID_2D)
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();
#endif /* GRID_2D */
#endif /* !GRID_1D */

    // Get pixel for image.
    FieldValue value = 0;
    switch (dump_type)
    {
      case CURRENT:
      {
        value = current->getCurValue ();
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        value = current->getPrevValue ();
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        value = current->getPrevPrevValue ();
        break;
      }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
      default:
      {
        UNREACHABLE;
      }
    }
    RGBApixel pixel = getPixelFromValue (value, maxNeg, max);

    // Set pixel for current image.
    image.SetPixel(px, py, pixel);
  }

  // Write image to file.
  switch (dump_type)
  {
    case CURRENT:
    {
      std::string cur_bmp = cur + std::string (".bmp");
      image.WriteToFile(cur_bmp.c_str());
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_bmp = prev + std::string (".bmp");
      image.WriteToFile(prev_bmp.c_str());
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_bmp = prevPrev + std::string (".bmp");
      image.WriteToFile(prevPrev_bmp.c_str());
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
 * Dumper for 1D and 2D grids.
 */
void
BMPDumper::dumpFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const
{
  writeToFile (grid, sx, sy, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    writeToFile (grid, sx, sy, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    writeToFile (grid, sx, sy, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}


#if defined (GRID_1D)
void
BMPDumper::dump1D (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();

  grid_coord sx = size.getX ();
  std::cout << "Saving 1D to BMP image. Size: " << sx << "x1. " << std::endl;

  dumpFlat (grid, sx, 1);

  std::cout << "Saved. " << std::endl;
}
#endif /* GRID_1D */

#if defined (GRID_2D)
void
BMPDumper::dump2D (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Saving 2D to BMP image. Size: " << sx << "x" << sy << ". " << std::endl;

  dumpFlat (grid, sx, sy);

  std::cout << "Saved. " << std::endl;
}
#endif /* GRID_2D */

#if defined (GRID_3D)
void
BMPDumper::dump3D (Grid& grid) const
{
  ASSERT_MESSAGE ("3D Dumper is not implemented.")
}
#endif /* GRID_3D */

#include <iostream>

#include "BMPDumper.h"

void
BMPDumper::dumpGrid (Grid& grid) const
{
#if defined (GRID_1D)
  dump1D (grid);
#else
#if defined (GRID_2D)
  dump2D (grid);
#else
#if defined (GRID_3D)
  dump3D (grid);
#endif
#endif
#endif
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

/**
 * Dumper for 1D and 2D grids.
 */
void
BMPDumper::dumpFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const
{
  // Create image for current values and max/min values.
  BMP imageCur;
  imageCur.SetSize (sx, sy);
  imageCur.SetBitDepth (24);

  FieldValue maxPosCur = grid.getValues ()[0].getCurValue ();
  FieldValue maxNegCur = grid.getValues ()[0].getCurValue ();

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // Create image for previous values and max/min values.
  BMP imagePrev;
  imagePrev.SetSize (sx, sy);
  imagePrev.SetBitDepth (24);

  FieldValue maxPosPrev = grid.getValues ()[0].getPrevValue ();
  FieldValue maxNegPrev = grid.getValues ()[0].getPrevValue ();

#if defined (TWO_TIME_STEPS)
  // Create image for previous previous values and max/min values.
  BMP imagePrevPrev;
  imagePrevPrev.SetSize (sx, sy);
  imagePrevPrev.SetBitDepth (24);

  FieldValue maxPosPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
  FieldValue maxNegPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
#endif
#endif


  // Go through all values and calculate max/min.
  for (FieldPointValue& current : grid.getValues ())
  {
    // Calculate max/min values for current values.
    const FieldValue& cur = current.getCurValue ();
    if (cur > maxPosCur)
    {
      maxPosCur = cur;
    }
    if (cur < maxNegCur)
    {
      maxNegCur = cur;
    }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    // Calculate max/min values for previous values.
    if (type == ALL)
    {
      const FieldValue& prev = current.getPrevValue ();
      if (prev > maxPosPrev)
      {
        maxPosPrev = prev;
      }
      if (prev < maxNegPrev)
      {
        maxNegPrev = prev;
      }
    }

#if defined (TWO_TIME_STEPS)
    // Calculate max/min values for previous previous values.
    if (type == ALL)
    {
      const FieldValue& prevPrev = current.getPrevPrevValue ();
      if (prevPrev > maxPosPrevPrev)
      {
        maxPosPrevPrev = prevPrev;
      }
      if (prevPrev < maxNegPrevPrev)
      {
        maxNegPrevPrev = prevPrev;
      }
    }
#endif
#endif
  }


  // Set max (diff between max positive and max negative).
  const FieldValue maxCur = maxPosCur - maxNegCur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  const FieldValue maxPrev = maxPosPrev - maxNegPrev;
#if defined (TWO_TIME_STEPS)
  const FieldValue maxPrevPrev = maxPosPrevPrev - maxNegPrevPrev;
#endif
#endif


  // Go through all values and set pixels.
  VectorFieldPointValues& values = grid.getValues ();
  grid_iter end = values.size ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    // Get current point value.
    FieldPointValue& current = values[iter];
    // Calculate its position from index in array.
    GridCoordinate coord = grid.calculatePositionFromIndex (iter);

    // Pixel coordinate.
#if defined (GRID_1D)
    grid_iter px = coord.getX ();
    grid_iter py = 0;
#else
#if defined (GRID_2D)
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();
#endif
#endif

    // Get pixel for current image.
    const FieldValue& cur = current.getCurValue ();
    RGBApixel pixelCur = getPixelFromValue (cur, maxNegCur, maxCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    // Get pixel for previous image.
    const FieldValue& prev = current.getPrevValue ();
    RGBApixel pixelPrev = getPixelFromValue (prev, maxNegPrev, maxPrev);
#if defined (TWO_TIME_STEPS)
    // Get pixel for previous previous image.
    const FieldValue& prevPrev = current.getPrevPrevValue ();
    RGBApixel pixelPrevPrev = getPixelFromValue (prevPrev, maxNegPrevPrev, maxPrevPrev);
#endif
#endif

    // Set pixel for current image.
    imageCur.SetPixel(px, py, pixelCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    // Set pixel for previous image.
    if (type == ALL)
    {
      imagePrev.SetPixel(px, py, pixelPrev);
    }
#if defined (TWO_TIME_STEPS)
    // Set pixel for previous previous image.
    if (type == ALL)
    {
      imagePrevPrev.SetPixel(px, py, pixelPrevPrev);
    }
#endif
#endif
  }


  // Write images to files.
  std::string cur_bmp = cur + std::string (".bmp");
  imageCur.WriteToFile(cur_bmp.c_str());
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    std::string prev_bmp = prev + std::string (".bmp");
    imagePrev.WriteToFile(prev_bmp.c_str());
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    std::string prevPrev_bmp = prevPrev + std::string (".bmp");
    imagePrevPrev.WriteToFile(prevPrev_bmp.c_str());
  }
#endif
#endif
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
#endif

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
#endif

#if defined (GRID_3D)
void
BMPDumper::dump3D (Grid& grid) const
{
  std::cout << "Not implemented." << std::endl;
}
#endif

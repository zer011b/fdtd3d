#include <iostream>
#include <fstream>
#include <sstream>

#include "BMPDumper.h"

void
BMPDumper::dumpGrid (Grid& grid)
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

RGBApixel
BMPDumper::setPixel (const FieldValue& val, const FieldValue& maxNeg,
                     const FieldValue& max)
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  double value = val - maxNeg;
  if (value > max / 2.0)
  {
    value -= max / 2;
    FieldValue tmp = 2 * value / max;
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
      tmp = 2 * value / max;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * 255;
    pixel.Blue = (1.0 - tmp) * 255;
  }

  return pixel;
}

void
BMPDumper::dumpFlat (Grid& grid, grid_iter sx, grid_iter sy)
{
// Create image for current values and max/min values
  BMP imageCur;
  imageCur.SetSize (sx, sy);
  imageCur.SetBitDepth (24);

  FieldValue maxPosCur = grid.getValues ()[0].getCurValue ();
  FieldValue maxNegCur = grid.getValues ()[0].getCurValue ();

  // Create image for previous values and max/min values
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  BMP imagePrev;
  imagePrev.SetSize (sx, sy);
  imagePrev.SetBitDepth (24);

  FieldValue maxPosPrev = grid.getValues ()[0].getPrevValue ();
  FieldValue maxNegPrev = grid.getValues ()[0].getPrevValue ();

  // Create image for previous previous values and max/min values
#if defined (TWO_TIME_STEPS)
  BMP imagePrevPrev;
  imagePrevPrev.SetSize (sx, sy);
  imagePrevPrev.SetBitDepth (24);

  FieldValue maxPosPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
  FieldValue maxNegPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
#endif
#endif

  // Calculate max values
  for (FieldPointValue& current : grid.getValues ())
  {
    FieldValue& cur = current.getCurValue ();
    if (cur > maxPosCur)
    {
      maxPosCur = cur;
    }
    if (cur < maxNegCur)
    {
      maxNegCur = cur;
    }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue& prev = current.getPrevValue ();
    if (prev > maxPosPrev)
    {
      maxPosPrev = prev;
    }
    if (prev < maxNegPrev)
    {
      maxNegPrev = prev;
    }

#if defined (TWO_TIME_STEPS)
    FieldValue& prevPrev = current.getPrevPrevValue ();
    if (prevPrev > maxPosPrevPrev)
    {
      maxPosPrevPrev = prevPrev;
    }
    if (prevPrev < maxNegPrevPrev)
    {
      maxNegPrevPrev = prevPrev;
    }
#endif
#endif
  }

  FieldValue maxCur = maxPosCur - maxNegCur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue maxPrev = maxPosPrev - maxNegPrev;
#if defined (TWO_TIME_STEPS)
  FieldValue maxPrevPrev = maxPosPrevPrev - maxNegPrevPrev;
#endif
#endif

  VectorFieldPointValues& values = grid.getValues ();
  grid_iter end = values.size ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    FieldPointValue& current = values[iter];
    GridCoordinate coord = grid.calculatePositionFromIndex (iter);

#if defined (GRID_1D)
    grid_iter px = coord.getX ();
    grid_iter py = 0;
#else
#if defined (GRID_2D)
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();
#endif
#endif

    FieldValue& cur = current.getCurValue ();
    RGBApixel pixelCur = setPixel (cur, maxNegCur, maxCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue& prev = current.getPrevValue ();
    RGBApixel pixelPrev = setPixel (prev, maxNegPrev, maxPrev);
#if defined (TWO_TIME_STEPS)
    FieldValue& prevPrev = current.getPrevPrevValue ();
    RGBApixel pixelPrevPrev = setPixel (prevPrev, maxNegPrevPrev, maxPrevPrev);
#endif
#endif

    imageCur.SetPixel(px, py, pixelCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    imagePrev.SetPixel(px, py, pixelPrev);
#if defined (TWO_TIME_STEPS)
    imagePrevPrev.SetPixel(px, py, pixelPrevPrev);
#endif
#endif
  }

  std::stringstream cur;
  cur << "cur.bmp";
  imageCur.WriteToFile(cur.str().c_str());
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  std::stringstream prev;
  prev << "prev.bmp";
  imagePrev.WriteToFile(prev.str().c_str());
#if defined (TWO_TIME_STEPS)
  std::stringstream prevPrev;
  prevPrev << "prevPrev.bmp";
  imagePrevPrev.WriteToFile(prevPrev.str().c_str());
#endif
#endif
}

#if defined (GRID_1D)
void
BMPDumper::dump1D (Grid& grid)
{
  GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();

  std::cout << "Saving 1D to BMP image. Size: " << sx << "x1. " << std::endl;

  dumpFlat (grid, sx, 1);

  std::cout << "Saved. " << std::endl;
}
#endif

#if defined (GRID_2D)
void
BMPDumper::dump2D (Grid& grid)
{
  GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Saving 2D to BMP image. Size: " << sx << "x" << sy << ". " << std::endl;

  dumpFlat (grid, sx, sy);

  std::cout << "Saved. " << std::endl;
}
#endif

#if defined (GRID_3D)
void
BMPDumper::dump3D (Grid& grid)
{

}
#endif
#include <iostream>

#include "BMPLoader.h"

void
BMPLoader::loadGrid (Grid& grid) const
{
  #if defined (GRID_1D)
    load1D (grid);
  #else /* GRID_1D */
  #if defined (GRID_2D)
    load2D (grid);
  #else /* GRID_2D */
  #if defined (GRID_3D)
    load3D (grid);
  #endif /* GRID_3D */
  #endif /* !GRID_2D */
  #endif /* !GRID_1D */
}

/*
 * Return value according to colors of pixel.
 * Blue-Green-Red scheme.
 */
FieldValue
BMPLoader::getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                              const FieldValue& max) const
{
  FieldValue retval = 0;
  FieldValue max_2 = max / 2.0;

  if (pixel.Blue == 0 && pixel.Red == 0)
  {
    retval = max_2;
  }
  else if (pixel.Blue == 0)
  {
    retval = (((FieldValue) pixel.Red) / 255 + 1) * max_2;
  }
  else if (pixel.Red == 0)
  {
    retval = (((FieldValue) pixel.Green) / 255) * max_2;
  }
  else
  {
    UNREACHABLE;
  }

  return retval;
}

void
BMPLoader::loadFromFile (Grid& grid, const grid_iter& sx, const grid_iter& sy, GridFileType load_type) const
{
  // Create image for values and max/min values.
  BMP image;
  image.SetSize (sx, sy);
  image.SetBitDepth (24);

  FieldValue max = 0;
  FieldValue maxNeg = 0;
  switch (load_type)
  {
    case CURRENT:
    {
      max = maxValuePos.getCurValue () - maxValueNeg.getCurValue ();
      maxNeg = maxValueNeg.getCurValue ();

      std::string cur_bmp = cur + std::string (".bmp");
      image.ReadFromFile (cur_bmp.c_str());
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      max = maxValuePos.getPrevValue () - maxValueNeg.getPrevValue ();
      maxNeg = maxValueNeg.getPrevValue ();

      std::string prev_bmp = prev + std::string (".bmp");
      image.ReadFromFile (prev_bmp.c_str());
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      max = maxValuePos.getPrevPrevValue () - maxValueNeg.getPrevPrevValue ();
      maxNeg = maxValueNeg.getPrevPrevValue ();

      std::string prevPrev_bmp = prevPrev + std::string (".bmp");
      image.ReadFromFile (prevPrev_bmp.c_str());
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
#endif /* GRID_1D */

    RGBApixel pixel = image.GetPixel(px, py);

    // Get pixel for image.
    FieldValue currentVal = getValueFromPixel (pixel, maxNeg, max);
    switch (load_type)
    {
      case CURRENT:
      {
        current->setCurValue (currentVal);
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        current->setPrevValue (currentVal);
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        current->setPrevPrevValue (currentVal);
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

void
BMPLoader::loadFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const
{
  loadFromFile (grid, sx, sy, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, sx, sy, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, sx, sy, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

#if defined (GRID_1D)
void
BMPLoader::load1D (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();

  std::cout << "Loading 1D from BMP image. Size: " << sx << "x1. " << std::endl;

  loadFlat (grid, sx, 1);

  std::cout << "Loaded. " << std::endl;
}
#endif /* GRID_1D */

#if defined (GRID_2D)
void
BMPLoader::load2D (Grid& grid) const
{
  const GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Loading 2D from BMP image. Size: " << sx << "x" << sy << ". " << std::endl;

  loadFlat (grid, sx, sy);

  std::cout << "Loaded. " << std::endl;
}
#endif /* GRID_2D */

#if defined (GRID_3D)
void
BMPLoader::load3D (Grid& grid) const
{

}
#endif /* GRID_3D */

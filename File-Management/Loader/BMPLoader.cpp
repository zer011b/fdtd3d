#include <iostream>

#include "BMPLoader.h"

template<>
void
BMPLoader<Grid1D>::loadGrid (Grid1D& grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate1D& size = grid.getSize ();
  grid_coord sx = size.getX ();

  std::cout << "Loading 1D from BMP image. Size: " << sx << "x1. " << std::endl;
#endif

  loadFromFile (grid);

#if PRINT_MESSAGE
  std::cout << "Loaded. " << std::endl;
#endif
}

template<>
void
BMPLoader<Grid2D>::loadGrid (Grid2D& grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate2D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Loading 2D from BMP image. Size: " << sx << "x" << sy << ". " << std::endl;
#endif

  loadFromFile (grid);

#if PRINT_MESSAGE
  std::cout << "Loaded. " << std::endl;
#endif
}

template<>
void
BMPLoader<Grid3D>::loadGrid (Grid3D& grid) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

template<>
void
BMPLoader<Grid1D>::loadFromFile (Grid1D& grid, GridFileType load_type)
{
  const GridCoordinate1D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = 1;

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
    grid_iter px = coord.getX ();
    grid_iter py = 0;

    RGBApixel pixel = image.GetPixel(px, py);

    // Get pixel for image.
    FieldValue currentVal = BMPhelper.getValueFromPixel (pixel, maxNeg, max);
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

template<>
void
BMPLoader<Grid2D>::loadFromFile (Grid2D& grid, GridFileType load_type)
{
  const GridCoordinate2D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

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
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();

    RGBApixel pixel = image.GetPixel(px, py);

    // Get pixel for image.
    FieldValue currentVal = BMPhelper.getValueFromPixel (pixel, maxNeg, max);
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

template<>
void
BMPLoader<Grid3D>::loadFromFile (Grid2D& grid, GridFileType load_type)
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

template<>
void
BMPLoader<Grid1D>::loadFromFile (Grid1D& grid)
{
  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

template<>
void
BMPLoader<Grid2D>::loadFromFile (Grid2D& grid)
{
  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

template<>
void
BMPLoader<Grid3D>::loadFromFile (Grid3D& grid)
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

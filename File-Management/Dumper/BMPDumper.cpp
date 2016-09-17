#include <iostream>

#include "BMPDumper.h"

/**
 * Virtual method for grid saving for 1D
 */
template<>
void
BMPDumper<GridCoordinate1D>::dumpGrid (Grid<GridCoordinate1D> &grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate1D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  std::cout << "Saving 1D to BMP image. Size: " << sx << "x1. " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Virtual method for grid saving for 2D
 */
template<>
void
BMPDumper<GridCoordinate2D>::dumpGrid (Grid<GridCoordinate2D> &grid) const
{
#if PRINT_MESSAGE
  const GridCoordinate2D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();

  std::cout << "Saving 2D to BMP image. Size: " << sx << "x" << sy << ". " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Virtual method for grid saving for 3D
 */
template<>
void
BMPDumper<GridCoordinate3D>::dumpGrid (Grid<GridCoordinate3D> &grid) const
{
//  ASSERT_MESSAGE ("3D Dumper is not implemented.")
#if PRINT_MESSAGE
  const GridCoordinate3D& size = grid.getSize ();

  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();
  grid_coord sz = size.getZ ();

  std::cout << "Saving 3D to BMP image. Size: " << sx << "x" << sy << "x" << sz << ". " << std::endl;
#endif /* PRINT_MESSAGE */

  writeToFile (grid);

#if PRINT_MESSAGE
  std::cout << "Saved. " << std::endl;
#endif /* PRINT_MESSAGE */
}

/**
 * Save grid to file for specific layer for 1D.
 */
template<>
void
BMPDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> &grid, GridFileType dump_type) const
{
  const GridCoordinate1D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = 1;

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
    GridCoordinate1D coord = grid.calculatePositionFromIndex (iter);

    // Pixel coordinate.
    grid_iter px = coord.getX ();
    grid_iter py = 0;

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
    RGBApixel pixel = BMPhelper.getPixelFromValue (value, maxNeg, max);

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
 * Save grid to file for specific layer for 2D.
 */
template<>
void
BMPDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> &grid, GridFileType dump_type) const
{
  const GridCoordinate2D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();;

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
    GridCoordinate2D coord = grid.calculatePositionFromIndex (iter);

    // Pixel coordinate.
    grid_iter px = coord.getX ();
    grid_iter py = coord.getY ();;

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
    RGBApixel pixel = BMPhelper.getPixelFromValue (value, maxNeg, max);

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
 * Save grid to file for specific layer for 3D.
 */
template<>
void
BMPDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> &grid, GridFileType dump_type) const
{
//  ASSERT_MESSAGE ("3D Dumper is not implemented.")

  const GridCoordinate3D& size = grid.getSize ();
  grid_coord sx = size.getX ();
  grid_coord sy = size.getY ();
  grid_coord sz = size.getZ ();

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

  // Create image for current values and max/min values.
  for (grid_coord k = 0; k < sz; ++k)
  {
    BMP image;
    image.SetSize (sx, sy);
    image.SetBitDepth (24);

    for (grid_iter i = 0; i < size.getX (); ++i)
    {
      for (grid_iter j = 0; j < size.getY (); ++j)
      {
        GridCoordinate3D pos (i, j, k);

        // Get current point value.
        const FieldPointValue* current = grid.getFieldPointValue (pos);
        ASSERT (current);

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
        RGBApixel pixel = BMPhelper.getPixelFromValue (value, maxNeg, max);

        // Set pixel for current image.
        image.SetPixel(i, j, pixel);
      }
    }

    // Write image to file.
    switch (dump_type)
    {
      case CURRENT:
      {
        std::string cur_bmp = cur + int64_to_string (k) + std::string (".bmp");
        image.WriteToFile(cur_bmp.c_str());
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        std::string prev_bmp = prev + int64_to_string (k) + std::string (".bmp");
        image.WriteToFile(prev_bmp.c_str());
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        std::string prevPrev_bmp = prevPrev + int64_to_string (k) + std::string (".bmp");
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
}

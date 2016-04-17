#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include <iostream>

#include "BMPHelper.h"
#include "Loader.h"

// Grid loader from BMP files.
template <class TCoord>
class BMPLoader: public Loader<TCoord>
{
  // Maximum positive value in grid.
  FieldPointValue maxValuePos;
  // Maximum negative value in grid.
  FieldPointValue maxValueNeg;

  static BMPHelper BMPhelper;

private:

  void loadFromFile (Grid<TCoord> &grid, GridFileType load_type) const;
  void loadFromFile (Grid<TCoord> &grid) const;

public:

  virtual ~BMPLoader () {}

  // Function to call for every grid type.
  virtual void loadGrid (Grid<TCoord> &grid) const override;

  // Setter and getter for maximum positive value.
  void setMaxValuePos (FieldPointValue& value)
  {
    maxValuePos = value;
  }
  const FieldPointValue& getMaxValuePos () const
  {
    return maxValuePos;
  }

  // Setter and getter for maximum negative value.
  void setMaxValueNeg (FieldPointValue& value)
  {
    maxValueNeg = value;
  }
  const FieldPointValue& getMaxValueNeg () const
  {
    return maxValueNeg;
  }
};

/**
 * Template implementation
 */

template<>
void
BMPLoader<GridCoordinate1D>::loadGrid (Grid<GridCoordinate1D> &grid) const
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
BMPLoader<GridCoordinate2D>::loadGrid (Grid<GridCoordinate2D> &grid) const
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
BMPLoader<GridCoordinate3D>::loadGrid (Grid<GridCoordinate3D> &grid) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

template<>
void
BMPLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> &grid, GridFileType load_type) const
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
    GridCoordinate1D coord = grid.calculatePositionFromIndex (iter);

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
BMPLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> &grid, GridFileType load_type) const
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
    GridCoordinate2D coord = grid.calculatePositionFromIndex (iter);

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
BMPLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> &grid, GridFileType load_type) const
{
  ASSERT_MESSAGE ("3D loader is not implemented.")
}

template <class TCoord>
void
BMPLoader<TCoord>::loadFromFile (Grid<TCoord> &grid) const
{
  loadFromFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    loadFromFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

#endif /* BMP_LOADER_H */

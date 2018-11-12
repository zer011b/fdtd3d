#include <iomanip>
#include <limits>

#include "TXTDumper.h"

/**
 * Save grid to file for specific layer.
 */
template <>
void
TXTDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> *grid,
                                          GridCoordinate1D startCoord,
                                          GridCoordinate1D endCoord,
                                          int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_1D (0, startCoord.getType1 ()) && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_1D (0, startCoord.getType1 ()) && endCoord <= grid->getSize ());

  std::ofstream file;
  file.open (names[time_step_back].c_str (), std::ios::out);
  ASSERT (file.is_open());
  
  file << std::setprecision (std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());
    grid_coord coord = grid->calculateIndexFromPosition (pos);

    file << pos.get1 () << " ";
    
    FieldValue *val = grid->getFieldValue (coord, time_step_back);
    
#ifdef COMPLEX_FIELD_VALUES
    file << val->real () << " " << val->imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
    file << *val << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
  }

  file.close();
}

template <>
void
TXTDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> *grid,
                                          GridCoordinate2D startCoord,
                                          GridCoordinate2D endCoord,
                                          int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && endCoord <= grid->getSize ());

  std::ofstream file;
  file.open (names[time_step_back].c_str (), std::ios::out);
  ASSERT (file.is_open());
  
  file << std::setprecision (std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());
      grid_coord coord = grid->calculateIndexFromPosition (pos);

      file << pos.get1 () << " " << pos.get2 () << " ";
      
      FieldValue *val = grid->getFieldValue (coord, time_step_back);
      
#ifdef COMPLEX_FIELD_VALUES
      file << val->real () << " " << val->imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
      file << *val << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
    }
  }

  file.close();
}

template <>
void
TXTDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> *grid,
                                          GridCoordinate3D startCoord,
                                          GridCoordinate3D endCoord,
                                          int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && endCoord <= grid->getSize ());

  std::ofstream file;
  file.open (names[time_step_back].c_str (), std::ios::out);
  ASSERT (file.is_open());
  
  file << std::setprecision (std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      for (grid_coord k = startCoord.get3 (); k < endCoord.get3 (); ++k)
      {
        GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k,
                                                   grid->getSize ().getType1 (),
                                                   grid->getSize ().getType2 (),
                                                   grid->getSize ().getType3 ());
        grid_coord coord = grid->calculateIndexFromPosition (pos);

        file << pos.get1 () << " " << pos.get2 () << " " << pos.get3 () << " ";
        
        FieldValue *val = grid->getFieldValue (coord, time_step_back);
        
#ifdef COMPLEX_FIELD_VALUES
        file << val->real () << " " << val->imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << *val << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
      }
    }
  }

  file.close();
}
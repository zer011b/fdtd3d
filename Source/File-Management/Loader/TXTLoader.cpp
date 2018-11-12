#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "TXTLoader.h"

/**
 * Virtual method for grid loading for 1D
 */
template<>
void
TXTLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid,
                                           GridCoordinate1D startCoord,
                                           GridCoordinate1D endCoord,
                                           int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_1D (0, startCoord.getType1 ()) && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_1D (0, startCoord.getType1 ()) && endCoord <= grid->getSize ());
  
  std::ifstream file;
  file.open (names[time_step_back].c_str (), std::ios::in);
  ASSERT (file.is_open());

  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    GridCoordinate1D pos = GRID_COORDINATE_1D (i, grid->getSize ().getType1 ());
    grid_coord coord = grid->calculateIndexFromPosition (pos);

    std::string line;

    std::getline (file, line);
    ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

    std::string buf;
    std::vector<std::string> tokens;
    std::stringstream ss (line);
    while (ss >> buf)
    {
      tokens.push_back(buf);
    }

    uint32_t word_index = 0;

    ASSERT (i == STOI (tokens[word_index].c_str ()));
    ++word_index;
    
    FPValue real = STOF (tokens[word_index++].c_str ());
    ASSERT (word_index == 2);
#ifdef COMPLEX_FIELD_VALUES
    FPValue imag = STOF (tokens[word_index++].c_str ());
    ASSERT (word_index == 3);
    grid->setFieldValue (FieldValue (real, imag), coord, time_step_back);
#else
    grid->setFieldValue (FieldValue (real), coord, time_step_back);
#endif
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

/**
 * Virtual method for grid loading for 2D
 */
template<>
void
TXTLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> *grid,
                                           GridCoordinate2D startCoord,
                                           GridCoordinate2D endCoord,
                                           int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_2D (0, 0, startCoord.getType1 (), startCoord.getType2 ())
          && endCoord <= grid->getSize ());
  
  std::ifstream file;
  file.open (names[time_step_back].c_str (), std::ios::in);
  ASSERT (file.is_open());

  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = startCoord.get1 (); i < endCoord.get1 (); ++i)
  {
    for (grid_coord j = startCoord.get2 (); j < endCoord.get2 (); ++j)
    {
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j, grid->getSize ().getType1 (), grid->getSize ().getType2 ());
      grid_coord coord = grid->calculateIndexFromPosition (pos);

      std::string line;

      std::getline (file, line);
      ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

      std::string buf;
      std::vector<std::string> tokens;
      std::stringstream ss (line);
      while (ss >> buf)
      {
        tokens.push_back(buf);
      }

      uint32_t word_index = 0;

      ASSERT (i == STOI (tokens[word_index].c_str ()));
      ++word_index;
      ASSERT (j == STOI (tokens[word_index].c_str ()));
      ++word_index;
    
      FPValue real = STOF (tokens[word_index++].c_str ());
      ASSERT (word_index == 3);
#ifdef COMPLEX_FIELD_VALUES
      FPValue imag = STOF (tokens[word_index++].c_str ());
      ASSERT (word_index == 4);
      grid->setFieldValue (FieldValue (real, imag), coord, time_step_back);
#else
      grid->setFieldValue (FieldValue (real), coord, time_step_back);
#endif
    }
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

/**
 * Virtual method for grid loading for 3D
 */
template<>
void
TXTLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> *grid,
                                           GridCoordinate3D startCoord,
                                           GridCoordinate3D endCoord,
                                           int time_step_back)
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && startCoord < grid->getSize ());
  ASSERT (endCoord > GRID_COORDINATE_3D (0, 0, 0, startCoord.getType1 (), startCoord.getType2 (), startCoord.getType3 ())
          && endCoord <= grid->getSize ());
  
  std::ifstream file;
  file.open (names[time_step_back].c_str (), std::ios::in);
  ASSERT (file.is_open());

  file >> std::setprecision(std::numeric_limits<double>::digits10);

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

        std::string line;

        std::getline (file, line);
        ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

        std::string buf;
        std::vector<std::string> tokens;
        std::stringstream ss (line);
        while (ss >> buf)
        {
          tokens.push_back(buf);
        }

        uint32_t word_index = 0;

        ASSERT (i == STOI (tokens[word_index].c_str ()));
        ++word_index;
        ASSERT (j == STOI (tokens[word_index].c_str ()));
        ++word_index;
        ASSERT (k == STOI (tokens[word_index].c_str ()));
        ++word_index;
    
        FPValue real = STOF (tokens[word_index++].c_str ());
        ASSERT (word_index == 4);
#ifdef COMPLEX_FIELD_VALUES
        FPValue imag = STOF (tokens[word_index++].c_str ());
        ASSERT (word_index == 5);
        grid->setFieldValue (FieldValue (real, imag), coord, time_step_back);
#else
        grid->setFieldValue (FieldValue (real), coord, time_step_back);
#endif
      }
    }
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

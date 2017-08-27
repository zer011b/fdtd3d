#include <iostream>

#include "TXTLoader.h"

/**
 * Virtual method for grid loading for 1D
 */
template<>
void
TXTLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = this->GridFileManager::cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::in);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = this->GridFileManager::prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::in);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = this->GridFileManager::prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::in);
      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (file.is_open());

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().getX (); ++i)
  {
    GridCoordinate1D pos (i);

    // Get current point value.
    FieldPointValue* current = grid->getFieldPointValue (pos);
    ASSERT (current);

    std::string line;

    file >> line;
    ASSERT (file.rdstate() & std::ifstream::failbit == 0);

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

    switch (type)
    {
      case CURRENT:
      {
        FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
        FPValue imag = STOF (tokens[word_index++].c_str ());
        current->setCurValue (FieldValue (real, imag));
#else
        current->setCurValue (FieldValue (real));
#endif
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
        FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
        FPValue imag = STOF (tokens[word_index++].c_str ());
        current->setPrevValue (FieldValue (real, imag));
#else
        current->setPrevValue (FieldValue (real));
#endif
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
        FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
        FPValue imag = STOF (tokens[word_index++].c_str ());
        current->setPrevPrevValue (FieldValue (real, imag));
#else
        current->setPrevPrevValue (FieldValue (real));
#endif
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

  ASSERT (file.eof());

  file.close();
}

/**
 * Virtual method for grid loading for 1D
 */
template<>
void
TXTLoader<GridCoordinate2D>::loadFromFile (Grid<GridCoordinate2D> *grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_dat = this->GridFileManager::cur + std::string (".dat");
      file.open (cur_dat.c_str (), std::ios::in);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_dat = this->GridFileManager::prev + std::string (".dat");
      file.open (prev_dat.c_str (), std::ios::in);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_dat = this->GridFileManager::prevPrev + std::string (".dat");
      file.open (prevPrev_dat.c_str (), std::ios::in);
      break;
    }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (file.is_open());

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().getX (); ++i)
  {
    for (grid_coord j = 0; j < grid->getSize ().getY (); ++j)
    {
      GridCoordinate2D pos (i, j);

      // Get current point value.
      FieldPointValue* current = grid->getFieldPointValue (pos);
      ASSERT (current);

      std::string line;

      file >> line;
      ASSERT (file.rdstate() & std::ifstream::failbit == 0);

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

      switch (type)
      {
        case CURRENT:
        {
          FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
          FPValue imag = STOF (tokens[word_index++].c_str ());
          current->setCurValue (FieldValue (real, imag));
#else
          current->setCurValue (FieldValue (real));
#endif
          break;
        }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        case PREVIOUS:
        {
          FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
          FPValue imag = STOF (tokens[word_index++].c_str ());
          current->setPrevValue (FieldValue (real, imag));
#else
          current->setPrevValue (FieldValue (real));
#endif
          break;
        }
#if defined (TWO_TIME_STEPS)
        case PREVIOUS2:
        {
          FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
          FPValue imag = STOF (tokens[word_index++].c_str ());
          current->setPrevPrevValue (FieldValue (real, imag));
#else
          current->setPrevPrevValue (FieldValue (real));
#endif
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

  ASSERT (file.eof());

  file.close();
}

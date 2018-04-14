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
TXTLoader<GridCoordinate1D>::loadFromFile (Grid<GridCoordinate1D> *grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::in);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_txt = this->GridFileManager::prev;
      if (!this->GridFileManager::manualFileNames)
      {
        prev_txt += std::string (".txt");
      }
      file.open (prev_txt.c_str (), std::ios::in);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_txt = this->GridFileManager::prevPrev;
      if (!this->GridFileManager::manualFileNames)
      {
        prevPrev_txt += std::string (".txt");
      }
      file.open (prevPrev_txt.c_str (), std::ios::in);
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
  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    GridCoordinate1D pos (i);

    // Get current point value.
    FieldPointValue* current = grid->getFieldPointValue (pos);
    ASSERT (current);

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

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

/**
 * Virtual method for grid loading for 2D
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
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::in);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_txt = this->GridFileManager::prev;
      if (!this->GridFileManager::manualFileNames)
      {
        prev_txt += std::string (".txt");
      }
      file.open (prev_txt.c_str (), std::ios::in);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_txt = this->GridFileManager::prevPrev;
      if (!this->GridFileManager::manualFileNames)
      {
        prevPrev_txt += std::string (".txt");
      }
      file.open (prevPrev_txt.c_str (), std::ios::in);
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
  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
      GridCoordinate2D pos (i, j);

      // Get current point value.
      FieldPointValue* current = grid->getFieldPointValue (pos);
      ASSERT (current);

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

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

/**
 * Virtual method for grid loading for 3D
 */
template<>
void
TXTLoader<GridCoordinate3D>::loadFromFile (Grid<GridCoordinate3D> *grid, GridFileType type) const
{
  std::ifstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::in);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
      std::string prev_txt = this->GridFileManager::prev;
      if (!this->GridFileManager::manualFileNames)
      {
        prev_txt += std::string (".txt");
      }
      file.open (prev_txt.c_str (), std::ios::in);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
      std::string prevPrev_txt = this->GridFileManager::prevPrev;
      if (!this->GridFileManager::manualFileNames)
      {
        prevPrev_txt += std::string (".txt");
      }
      file.open (prevPrev_txt.c_str (), std::ios::in);
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
  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        // Get current point value.
        FieldPointValue* current = grid->getFieldPointValue (pos);
        ASSERT (current);

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
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
}

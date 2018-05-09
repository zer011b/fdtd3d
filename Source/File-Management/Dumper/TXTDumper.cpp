#include <iomanip>
#include <limits>

#include "TXTDumper.h"

/**
 * Save grid to file for specific layer.
 */
template <>
void
TXTDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> *grid, GridFileType type, GridCoordinate1D startCoord, GridCoordinate1D endCoord) const
{
  /**
   * TODO: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::out);
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
      file.open (prev_txt.c_str (), std::ios::out);
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
      file.open (prevPrev_txt.c_str (), std::ios::out);
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
  file << std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , grid->getSize ().getType1 ()
#endif
                         );

    // Get current point value.
    const FieldPointValue* current = grid->getFieldPointValue (pos);
    ASSERT (current);

    file << pos.get1 () << " ";

    switch (type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getCurValue ().real () << " " << current->getCurValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getPrevValue ().real () << " " << current->getPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getPrevPrevValue ().real () << " " << current->getPrevPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << current->getPrevPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
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

  file.close();
}

template <>
void
TXTDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> *grid, GridFileType type, GridCoordinate2D startCoord, GridCoordinate2D endCoord) const
{
  /**
   * TODO: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::out);
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
      file.open (prev_txt.c_str (), std::ios::out);
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
      file.open (prevPrev_txt.c_str (), std::ios::out);
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
  file << std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
      GridCoordinate2D pos (i, j
#ifdef DEBUG_INFO
                            , grid->getSize ().getType1 ()
                            , grid->getSize ().getType2 ()
#endif
                           );

      // Get current point value.
      const FieldPointValue* current = grid->getFieldPointValue (pos);
      ASSERT (current);

      file << pos.get1 () << " " << pos.get2 () << " ";

      switch (type)
      {
        case CURRENT:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getCurValue ().real () << " " << current->getCurValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
          file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
          break;
        }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        case PREVIOUS:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getPrevValue ().real () << " " << current->getPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
          file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
          break;
        }
#if defined (TWO_TIME_STEPS)
        case PREVIOUS2:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getPrevPrevValue ().real () << " " << current->getPrevPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
          file << current->getPrevPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
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

  file.close();
}

template <>
void
TXTDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> *grid, GridFileType type, GridCoordinate3D startCoord, GridCoordinate3D endCoord) const
{
  /**
   * TODO: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
      std::string cur_txt = this->GridFileManager::cur;
      if (!this->GridFileManager::manualFileNames)
      {
        cur_txt += std::string (".txt");
      }
      file.open (cur_txt.c_str (), std::ios::out);
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
      file.open (prev_txt.c_str (), std::ios::out);
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
      file.open (prevPrev_txt.c_str (), std::ios::out);
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
  file << std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid->getSize ().get1 (); ++i)
  {
    for (grid_coord j = 0; j < grid->getSize ().get2 (); ++j)
    {
      for (grid_coord k = 0; k < grid->getSize ().get3 (); ++k)
      {
        GridCoordinate3D pos (i, j, k
#ifdef DEBUG_INFO
                              , grid->getSize ().getType1 ()
                              , grid->getSize ().getType2 ()
                              , grid->getSize ().getType3 ()
#endif
                             );

        // Get current point value.
        const FieldPointValue* current = grid->getFieldPointValue (pos);
        ASSERT (current);

        file << pos.get1 () << " " << pos.get2 () << " " << pos.get3 () << " ";

        switch (type)
        {
          case CURRENT:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getCurValue ().real () << " " << current->getCurValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
            file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
            break;
          }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          case PREVIOUS:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getPrevValue ().real () << " " << current->getPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
            file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
            break;
          }
#if defined (TWO_TIME_STEPS)
          case PREVIOUS2:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getPrevPrevValue ().real () << " " << current->getPrevPrevValue ().imag () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
            file << current->getPrevPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
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

  file.close();
}

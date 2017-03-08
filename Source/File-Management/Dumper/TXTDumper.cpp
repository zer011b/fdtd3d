#include "TXTDumper.h"

/**
 * Save grid to file for specific layer.
 */
template <>
void
TXTDumper<GridCoordinate1D>::writeToFile (Grid<GridCoordinate1D> &grid, GridFileType type, GridCoordinate1D startCoord, GridCoordinate1D endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
#ifdef CXX11_ENABLED
      std::string cur_txt = GridFileManager::cur + std::string (".txt");
#else
      std::string cur_txt = this->GridFileManager::cur + std::string (".txt");
#endif
      file.open (cur_txt.c_str (), std::ios::out);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef CXX11_ENABLED
      std::string prev_txt = GridFileManager::prev + std::string (".txt");
#else
      std::string prev_txt = this->GridFileManager::prev + std::string (".txt");
#endif
      file.open (prev_txt.c_str (), std::ios::out);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef CXX11_ENABLED
      std::string prevPrev_txt = GridFileManager::prevPrev + std::string (".txt");
#else
      std::string prevPrev_txt = this->GridFileManager::prevPrev + std::string (".txt");
#endif
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

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid.getSize ().getX (); ++i)
  {
    GridCoordinate1D pos (i);

    // Get current point value.
    const FieldPointValue* current = grid.getFieldPointValue (i);
    ASSERT (current);

    file << pos.getX () << " ";

    switch (type)
    {
      case CURRENT:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getCurValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
        break;
      }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      case PREVIOUS:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getPrevValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
        file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
        break;
      }
#if defined (TWO_TIME_STEPS)
      case PREVIOUS2:
      {
#ifdef COMPLEX_FIELD_VALUES
        file << current->getPrevPrevValue ().real () << std::endl;
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

    file << std::endl;
  }

  file.close();
}

template <>
void
TXTDumper<GridCoordinate2D>::writeToFile (Grid<GridCoordinate2D> &grid, GridFileType type, GridCoordinate2D startCoord, GridCoordinate2D endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
#ifdef CXX11_ENABLED
      std::string cur_txt = GridFileManager::cur + std::string (".txt");
#else
      std::string cur_txt = this->GridFileManager::cur + std::string (".txt");
#endif
      file.open (cur_txt.c_str (), std::ios::out);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef CXX11_ENABLED
      std::string prev_txt = GridFileManager::prev + std::string (".txt");
#else
      std::string prev_txt = this->GridFileManager::prev + std::string (".txt");
#endif
      file.open (prev_txt.c_str (), std::ios::out);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef CXX11_ENABLED
      std::string prevPrev_txt = GridFileManager::prevPrev + std::string (".txt");
#else
      std::string prevPrev_txt = this->GridFileManager::prevPrev + std::string (".txt");
#endif
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

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid.getSize ().getX (); ++i)
  {
    for (grid_coord j = 0; j < grid.getSize ().getY (); ++j)
    {
      GridCoordinate2D pos (i, j);

      // Get current point value.
      const FieldPointValue* current = grid.getFieldPointValue (pos);
      ASSERT (current);

      file << pos.getX () << " " << pos.getY () << " ";

      switch (type)
      {
        case CURRENT:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getCurValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
          file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
          break;
        }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
        case PREVIOUS:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getPrevValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
          file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
          break;
        }
#if defined (TWO_TIME_STEPS)
        case PREVIOUS2:
        {
#ifdef COMPLEX_FIELD_VALUES
          file << current->getPrevPrevValue ().real () << std::endl;
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

    file << std::endl;
  }

  file.close();
}

template <>
void
TXTDumper<GridCoordinate3D>::writeToFile (Grid<GridCoordinate3D> &grid, GridFileType type, GridCoordinate3D startCoord, GridCoordinate3D endCoord) const
{
  /**
   * FIXME: use startCoord and endCoord
   */
  std::ofstream file;
  switch (type)
  {
    case CURRENT:
    {
#ifdef CXX11_ENABLED
      std::string cur_txt = GridFileManager::cur + std::string (".txt");
#else
      std::string cur_txt = this->GridFileManager::cur + std::string (".txt");
#endif
      file.open (cur_txt.c_str (), std::ios::out);
      break;
    }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    case PREVIOUS:
    {
#ifdef CXX11_ENABLED
      std::string prev_txt = GridFileManager::prev + std::string (".txt");
#else
      std::string prev_txt = this->GridFileManager::prev + std::string (".txt");
#endif
      file.open (prev_txt.c_str (), std::ios::out);
      break;
    }
#if defined (TWO_TIME_STEPS)
    case PREVIOUS2:
    {
#ifdef CXX11_ENABLED
      std::string prevPrev_txt = GridFileManager::prevPrev + std::string (".txt");
#else
      std::string prevPrev_txt = this->GridFileManager::prevPrev + std::string (".txt");
#endif
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

  // Go through all values and write to file.
  for (grid_coord i = 0; i < grid.getSize ().getX (); ++i)
  {
    for (grid_coord j = 0; j < grid.getSize ().getY (); ++j)
    {
      for (grid_coord k = 0; k < grid.getSize ().getZ (); ++k)
      {
        GridCoordinate3D pos (i, j, k);

        // Get current point value.
        const FieldPointValue* current = grid.getFieldPointValue (pos);
        ASSERT (current);

        file << pos.getX () << " " << pos.getY () << " " << pos.getZ () << " ";

        switch (type)
        {
          case CURRENT:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getCurValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
            file << current->getCurValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
            break;
          }
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          case PREVIOUS:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getPrevValue ().real () << std::endl;
#else /* COMPLEX_FIELD_VALUES */
            file << current->getPrevValue () << std::endl;
#endif /* !COMPLEX_FIELD_VALUES */
            break;
          }
#if defined (TWO_TIME_STEPS)
          case PREVIOUS2:
          {
#ifdef COMPLEX_FIELD_VALUES
            file << current->getPrevPrevValue ().real () << std::endl;
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

    file << std::endl;
  }

  file.close();
}

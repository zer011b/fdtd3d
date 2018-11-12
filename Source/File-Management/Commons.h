#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "Assert.h"
#include "GridInterface.h"

extern std::string int64_to_string(int64_t value);

enum FileType
{
  FILE_TYPE_BMP,
  FILE_TYPE_DAT,
  FILE_TYPE_TXT,
  FILE_TYPE_COUNT
};

/**
 * Base class for all dumpers/loaders.
 */
class GridFileManager
{
protected:
  
  /**
   * Index of grid time step to save/load:
   *   -1:  save/load all time steps
   *   >=0: save/load specific time step
   */
  int index_of_grid;

  std::vector< std::string > names;

  void setFileNames (int savedSteps,
                     time_step step,
                     int processId,
                     const std::string & customName,
                     FileType ftype)
  {
    bool singleName = false;
    if (savedSteps == -1)
    {
      /*
       * Make only one file
       */
      savedSteps = 1;
      singleName = true;
    }
    
    ASSERT (savedSteps > 0);

    names.resize (savedSteps);

    for (int i = 0; i < names.size (); ++i)
    {
      if (singleName)
      {
        names[i] = std::string ("previous");
      }
      else
      {
        names[i] = std::string ("previous-") + int64_to_string (i);
      }
      
      names[i] += std::string ("_[timestep=") + int64_to_string (step)
                  + std::string ("]_[pid=") + int64_to_string (processId) + std::string ("]_[name=") + customName
                  + std::string ("]");

      switch (ftype)
      {
        case FILE_TYPE_BMP:
        {
          names[i] += std::string (".bmp");
          break;
        }
        case FILE_TYPE_DAT:
        {
          names[i] += std::string (".dat");
          break;
        }
        case FILE_TYPE_TXT:
        {
          names[i] += std::string (".txt");
          break;
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }
  }
  
  void setCustomFileNames (const std::vector< std::string > & customNames)
  {
    names.resize (customNames.size ());
    
    for (int i = 0; i < names.size (); ++i)
    {
      names[i] = customNames[i];
    }
  }
  
  /**
   * Protected constructor to save/load all/specific grid time step
   */
  GridFileManager () {}

public:

  virtual ~GridFileManager () {}

  static FileType getFileType (const std::string &);
};

#endif /* COMMONS_H */

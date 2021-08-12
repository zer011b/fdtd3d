/*
 * Copyright (C) 2015 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "Assert.h"
#include "GridInterface.h"

extern std::string int64_to_string(int64_t value);

/**
 * Type of saved/loaded file
 */
enum FileType
{
  FILE_TYPE_BMP, /**< BMP image */
  FILE_TYPE_DAT, /**< Binary file */
  FILE_TYPE_TXT, /**< Text file */
  FILE_TYPE_COUNT
}; /* FileType */

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

  /**
   * File names
   */
  std::vector< std::string > names;

  /**
   * Set file names
   */
  void setFileNames (int savedSteps, /**< number of saved files */
                     time_step step, /**< absolute time step */
                     int processId, /**< pid */
                     const std::string & customName, /**< custom file names */
                     FileType ftype) /**< type of file */
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
  } /* setFileNames */

  /**
   * Set custom file names
   */
  void setCustomFileNames (const std::vector< std::string > & customNames) /**< custom names */
  {
    names.resize (customNames.size ());

    for (int i = 0; i < names.size (); ++i)
    {
      names[i] = customNames[i];
    }
  } /* setCustomFileNames */

  /**
   * Protected constructor to save/load all/specific grid time step
   */
  GridFileManager () {}

public:

  virtual ~GridFileManager () {}

  static FileType getFileType (const std::string &);
}; /* GridFileManager */

#endif /* COMMONS_H */

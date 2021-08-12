/*
 * Copyright (C) 2017 Gleb Balykov
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

#ifndef TXT_LOADER_H
#define TXT_LOADER_H

#include <iostream>
#include <fstream>

#include <iomanip>
#include <limits>
#include <sstream>

#include "Loader.h"

/**
 * Grid loader from txt files.
 */
template <class TCoord>
class TXTLoader: public Loader<TCoord>
{
  void loadFromFile (Grid<TCoord> *, TCoord, TCoord, int);
  void loadGridInternal (Grid<TCoord> *, TCoord, TCoord, time_step, int);
  uint32_t skipIndexes (TCoord, const std::vector<std::string> &);

public:

  virtual ~TXTLoader () {}

  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
}; /* TXTLoader */

/**
 * Load data from file
 */
template<class TCoord>
void
TXTLoader<TCoord>::loadFromFile (Grid<TCoord> *grid, /**< grid to load */
                                 TCoord startCoord, /**< start loading from this coordinate */
                                 TCoord endCoord, /**< end loading at this coordinate */
                                 int time_step_back) /**< relative time step at which to load */
{
  ASSERT (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ());
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > endCoord.getZero () && endCoord <= grid->getSize ());

  std::ifstream file;
  file.open (this->GridFileManager::names[time_step_back].c_str (), std::ios::in);
  ASSERT (file.is_open());

  file >> std::setprecision(std::numeric_limits<double>::digits10);

  // Go through all values and write to file.
  typename VectorFieldValues<TCoord>::Iterator iter (startCoord, startCoord, endCoord);
  typename VectorFieldValues<TCoord>::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (startCoord, endCoord);
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();

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

    uint32_t word_index = skipIndexes (pos, tokens);

    FPValue real = STOF (tokens[word_index++].c_str ());
#ifdef COMPLEX_FIELD_VALUES
    FPValue imag = STOF (tokens[word_index++].c_str ());
    grid->setFieldValue (FieldValue (real, imag), pos, time_step_back);
#else /* COMPLEX_FIELD_VALUES */
    grid->setFieldValue (FieldValue (real), pos, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */
  }

  // peek next character from file, which should set eof flags
  ASSERT ((file.peek (), file.eof()));

  file.close();
} /* TXTLoader::loadFromFile */

/**
 * Choose scenario of loading of grid
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGridInternal (Grid<TCoord> *grid, /**< grid to load */
                                     TCoord startCoord, /**< start loading from this coordinate */
                                     TCoord endCoord, /**< end loading at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to load */
                                     int time_step_back) /**< relative time step at which to load */
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Loading grid '" << grid->getName () << "' from text. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Loading grid '" << grid->getName () << "' from text. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  if (time_step_back == -1)
  {
    /**
     * Save all time steps
     */
    for (int i = 0; i < grid->getCountStoredSteps (); ++i)
    {
      loadFromFile (grid, startCoord, endCoord, i);
    }
  }
  else
  {
    loadFromFile (grid, startCoord, endCoord, time_step_back);
  }

  std::cout << "Loaded. " << std::endl;
} /* TXTLoader::loadGridInternal */

/**
 * Virtual method for grid loading, which makes file names automatically
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             int pid) /**< pid of process, which does loading */
{
  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_TXT);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* TXTLoader::loadGrid */

/**
 * Virtual method for grid loading, which uses custom names
 */
template <class TCoord>
void
TXTLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* TXTLoader::loadGrid */

#endif /* TXT_LOADER_H */

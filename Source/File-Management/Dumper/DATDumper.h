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

#ifndef DAT_DUMPER_H
#define DAT_DUMPER_H

#include <iostream>
#include <fstream>

#include "Dumper.h"

/**
 * Grid saver to binary files.
 */
template <class TCoord>
class DATDumper: public Dumper<TCoord>
{
  void writeToFile (Grid<TCoord> *, TCoord, TCoord, int);
  void dumpGridInternal (Grid<TCoord> *, TCoord, TCoord, time_step, int);

public:

  virtual ~DATDumper () {}

  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;
}; /* DATDumper */

/**
 * Write data to file
 */
template <class TCoord>
void
DATDumper<TCoord>::writeToFile (Grid<TCoord> *grid, /**< grid to save */
                                TCoord startCoord, /**< start saving from this coordinate */
                                TCoord endCoord, /**< end saving at this coordinate */
                                int time_step_back) /**< relative time step at which to save */
{
  ASSERT ((time_step_back == -1) || (time_step_back >= 0 && time_step_back < grid->getCountStoredSteps ()));
  ASSERT (startCoord >= startCoord.getZero () && startCoord < grid->getSize ());
  ASSERT (endCoord > startCoord.getZero () && endCoord <= grid->getSize ());

  std::ofstream file;
  file.open (this->GridFileManager::names[time_step_back == -1 ? 0 : time_step_back].c_str (), std::ios::out | std::ios::binary);
  ASSERT (file.is_open());

  // Go through all values and write to file.
  typename VectorFieldValues<TCoord>::Iterator iter (startCoord, startCoord, endCoord);
  typename VectorFieldValues<TCoord>::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (startCoord, endCoord);
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();

    if (time_step_back == -1)
    {
      for (int i = 0; i < grid->getCountStoredSteps (); ++i)
      {
        file.write ((char*) (grid->getFieldValue (pos, i)), sizeof (FieldValue));
      }
    }
    else
    {
      file.write ((char*) (grid->getFieldValue (pos, time_step_back)), sizeof (FieldValue));
    }
  }

  file.close();
} /* DATDumper::writeToFile */

/**
 * Choose scenario of saving of grid
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, /**< grid to save */
                                     TCoord startCoord, /**< start saving from this coordinate */
                                     TCoord endCoord, /**< end saving at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to save */
                                     int time_step_back) /**< relative time step at which to save */
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Saving grid '" << grid->getName () << "' to binary. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Saving grid '" << grid->getName () << "' to binary. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  writeToFile (grid, startCoord, endCoord, time_step_back);

  std::cout << "Saved. " << std::endl;
} /* DATDumper::dumpGridInternal */

/**
 * Virtual method for grid saving, which makes file names automatically
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             int pid) /**< pid of process, which does saving */
{
  GridFileManager::setFileNames (time_step_back, timeStep, pid, std::string (grid->getName ()), FILE_TYPE_DAT);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* DATDumper::dumpGrid */

/**
 * Virtual method for grid saving, which uses custom names
 */
template <class TCoord>
void
DATDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* DATDumper::dumpGrid */

#endif /* DAT_DUMPER_H */

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

#ifndef DUMPER_H
#define DUMPER_H

#include "Commons.h"

/**
 * Base class for all dumpers.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class Dumper: public GridFileManager
{
public:

  virtual ~Dumper () {}

  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) = 0;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, const std::vector< std::string > &) = 0;
}; /* Dumper */

#endif /* DUMPER_H */

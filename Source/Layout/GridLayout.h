/*
 * Copyright (C) 2016 Gleb Balykov
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

#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"

/**
 * Direction in which to get circut elements
 */
ENUM_CLASS (LayoutDirection, uint8_t,
  LEFT, /**< left by Ox */
  RIGHT, /**< right by Ox */
  DOWN, /**< left by Oy */
  UP, /**< right by Oy */
  BACK, /**< left by Oz */
  FRONT /**< right by Oz */
);

/**
 * Type of electromagnetic field.
 */
ENUM_CLASS (GridType, uint8_t,
  NONE,
  EX,
  EY,
  EZ,
  HX,
  HY,
  HZ,
  DX,
  DY,
  DZ,
  BX,
  BY,
  BZ,
  EPS,
  MU,
  SIGMAX,
  SIGMAY,
  SIGMAZ,
  OMEGAPE,
  OMEGAPM,
  GAMMAE,
  GAMMAM
);

#endif /* GRID_LAYOUT_H */

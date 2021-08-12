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

#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include "FieldValue.h"
#include <cmath>

namespace PhysicsConst
{
  const CUDA_DEVICE FPValue Pi = M_PI;
  const CUDA_DEVICE FPValue SpeedOfLight = 299792458;
  const CUDA_DEVICE FPValue Mu0 = 4 * Pi * 0.0000001;
  const CUDA_DEVICE FPValue Eps0 = 1 / (Mu0 * SQR (SpeedOfLight));
  const CUDA_DEVICE FPValue accuracy = 0.001;
};

#endif /* PHYSICS_CONSTANTS_H */

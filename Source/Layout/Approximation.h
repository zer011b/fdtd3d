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

#ifndef APPROXIMATION_H
#define APPROXIMATION_H

#include "FieldValue.h"
#include "GridCoordinate3D.h"

#define APPROXIMATION_ACCURACY FPValue (0.0000001)

class Approximation
{
public:

  static CUDA_DEVICE CUDA_HOST FPValue getAccuracy ();

  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static CUDA_DEVICE CUDA_HOST FPValue getMaterial (const FieldValue &);

  static CUDA_DEVICE CUDA_HOST FPValue phaseVelocityIncidentWave (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateWaveNumber (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateWaveNumberGeneral (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FieldValue approximateSphereFast (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue);

  static FieldValue approximateSphereAccurate (GridCoordinateFP1D, GridCoordinateFP1D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereAccurate (GridCoordinateFP2D, GridCoordinateFP2D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereAccurate (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue, FieldValue);

  static FieldValue approximateSphereStair (GridCoordinateFP1D, GridCoordinateFP1D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereStair (GridCoordinateFP2D, GridCoordinateFP2D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereStair (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue, FieldValue);
};

#endif /* APPROXIMATION_H */

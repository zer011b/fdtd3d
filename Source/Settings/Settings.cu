/*
 * Copyright (C) 2018 Gleb Balykov
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

#define SETTINGS_CU
#define CUDA_SOURCES

#ifdef CUDA_ENABLED

#include "Settings.h"

__constant__ Settings *cudaSolverSettings;

#include "Settings.cpp"

CUDA_HOST
void
Settings::prepareDeviceSettings ()
{
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_cudaSolverSettings, sizeof (Settings)));
  cudaCheckErrorCmd (cudaMemcpy (d_cudaSolverSettings, this, sizeof (Settings), cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpyToSymbol (cudaSolverSettings, &d_cudaSolverSettings, sizeof (Settings*), 0, cudaMemcpyHostToDevice));

#ifdef ENABLE_ASSERTS
  Settings *d_tmp = NULLPTR;
  USED(d_tmp);
  cudaCheckErrorCmd (cudaMemcpyFromSymbol (&d_tmp, cudaSolverSettings, sizeof(Settings *), 0, cudaMemcpyDeviceToHost));
  Settings *tmp2 = (Settings *) malloc (sizeof (Settings));
  cudaCheckErrorCmd (cudaMemcpy (tmp2, d_tmp, sizeof(Settings), cudaMemcpyDeviceToHost));
  ALWAYS_ASSERT (tmp2->getCudaSettings () == solverSettings.getCudaSettings ());
  free (tmp2);
#endif /* ENABLE_ASSERTS */
}

CUDA_HOST
void
Settings::freeDeviceSettings ()
{
  cudaCheckErrorCmd (cudaFree (d_cudaSolverSettings));
}
#endif /* CUDA_ENABLED */

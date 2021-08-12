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

#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H

#ifdef CUDA_ENABLED
#ifdef CUDA_SOURCES

enum CudaExitStatus
{
  CUDA_OK,
  CUDA_ERROR
};

#ifdef __CUDA_ARCH__

#define cudaCheckError()
#define cudaCheckErrorCmd(cmd)
#define cudaCheckExitStatus(...)

#else /* __CUDA_ARCH__ */

#define cudaCheckError() \
  { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) \
    { \
      fprintf(stderr, "Fatal error: %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  }

#define cudaCheckErrorCmd(cmd) \
  { \
    (cmd); \
    cudaCheckError(); \
  }

#define cudaCheckExitStatus(...) \
  { \
    __VA_ARGS__; \
    cudaCheckErrorCmd (cudaMemcpy (&exitStatus, exitStatusCuda, sizeof (CudaExitStatus), cudaMemcpyDeviceToHost)); \
    if (exitStatus != CUDA_OK) \
    { \
      cudaCheckError (); \
      *retval = CUDA_ERROR; \
      return; \
    } \
  }

#endif /* !__CUDA_ARCH__ */

#endif /* CUDA_SOURCES */
#endif /* CUDA_ENABLED */

#endif /* !CUDA_INCLUDE_H */

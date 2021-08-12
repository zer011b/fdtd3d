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

#ifndef INTERNAL_SCHEME_H
#define INTERNAL_SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"

/**
 * InternalScheme is implemented without virtual functions in order to be copied to GPU (classes with vtable can't be)
 */

/*
 * Forward declaration of both CPU and GPU internal schemes
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class InternalScheme;
class InternalSchemeHelper;

#ifdef CUDA_ENABLED
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class InternalSchemeGPU;
class InternalSchemeHelperGPU;
#endif /* CUDA_ENABLED */

/*
 * ====================================
 * ======== CPU InternalScheme ========
 * ====================================
 */
#define INTERNAL_SCHEME_BASE InternalScheme
#define INTERNAL_SCHEME_HELPER InternalSchemeHelper
#define IGRID Grid
#define ICUDA_HOST CUDA_HOST
#define ICUDA_DEVICE

#include "InternalScheme.inc.h"

/*
 * ====================================
 * ======== GPU InternalScheme ========
 * ====================================
 */
#ifdef CUDA_ENABLED
#ifdef CUDA_SOURCES

#define GPU_INTERNAL_SCHEME

#define INTERNAL_SCHEME_BASE InternalSchemeGPU
#define INTERNAL_SCHEME_HELPER InternalSchemeHelperGPU
#define IGRID CudaGrid
#define ICUDA_HOST CUDA_HOST
#define ICUDA_DEVICE CUDA_DEVICE

#include "InternalScheme.inc.h"

#endif /* CUDA_SOURCES */
#endif /* CUDA_ENABLED */

#include "InternalScheme.specific.h"

#endif /* !INTERNAL_SCHEME_H */

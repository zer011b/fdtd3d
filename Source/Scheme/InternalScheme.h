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

#define GPU_INTERNAL_SCHEME

#define INTERNAL_SCHEME_BASE InternalSchemeGPU
#define INTERNAL_SCHEME_HELPER InternalSchemeHelperGPU
#define IGRID CudaGrid
#define ICUDA_HOST CUDA_HOST
#define ICUDA_DEVICE CUDA_DEVICE

#include "InternalScheme.inc.h"

#endif /* CUDA_ENABLED */

#include "InternalScheme.template.specific.h"

#endif /* !INTERNAL_SCHEME_H */

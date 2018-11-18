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

#define INTERNAL_SCHEME_1D InternalScheme1D
#define INTERNAL_SCHEME_2D InternalScheme2D
#define INTERNAL_SCHEME_3D InternalScheme3D

#define INTERNAL_SCHEME_1D_EX_HY InternalScheme1D_ExHy
#define INTERNAL_SCHEME_1D_EX_HZ InternalScheme1D_ExHz
#define INTERNAL_SCHEME_1D_EY_HX InternalScheme1D_EyHx
#define INTERNAL_SCHEME_1D_EY_HZ InternalScheme1D_EyHz
#define INTERNAL_SCHEME_1D_EZ_HX InternalScheme1D_EzHx
#define INTERNAL_SCHEME_1D_EZ_HY InternalScheme1D_EzHy

#define INTERNAL_SCHEME_2D_TEX InternalScheme2D_TEx
#define INTERNAL_SCHEME_2D_TEY InternalScheme2D_TEy
#define INTERNAL_SCHEME_2D_TEZ InternalScheme2D_TEz
#define INTERNAL_SCHEME_2D_TMX InternalScheme2D_TMx
#define INTERNAL_SCHEME_2D_TMY InternalScheme2D_TMy
#define INTERNAL_SCHEME_2D_TMZ InternalScheme2D_TMz

#define INTERNAL_SCHEME_3D_3D InternalScheme3D_3D


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

#define INTERNAL_SCHEME_1D InternalSchemeGPU1D
#define INTERNAL_SCHEME_2D InternalSchemeGPU2D
#define INTERNAL_SCHEME_3D InternalSchemeGPU3D

#define INTERNAL_SCHEME_1D_EX_HY InternalSchemeGPU1D_ExHy
#define INTERNAL_SCHEME_1D_EX_HZ InternalSchemeGPU1D_ExHz
#define INTERNAL_SCHEME_1D_EY_HX InternalSchemeGPU1D_EyHx
#define INTERNAL_SCHEME_1D_EY_HZ InternalSchemeGPU1D_EyHz
#define INTERNAL_SCHEME_1D_EZ_HX InternalSchemeGPU1D_EzHx
#define INTERNAL_SCHEME_1D_EZ_HY InternalSchemeGPU1D_EzHy

#define INTERNAL_SCHEME_2D_TEX InternalSchemeGPU2D_TEx
#define INTERNAL_SCHEME_2D_TEY InternalSchemeGPU2D_TEy
#define INTERNAL_SCHEME_2D_TEZ InternalSchemeGPU2D_TEz
#define INTERNAL_SCHEME_2D_TMX InternalSchemeGPU2D_TMx
#define INTERNAL_SCHEME_2D_TMY InternalSchemeGPU2D_TMy
#define INTERNAL_SCHEME_2D_TMZ InternalSchemeGPU2D_TMz

#define INTERNAL_SCHEME_3D_3D InternalSchemeGPU3D_3D

#include "InternalScheme.inc.h"

#endif /* CUDA_ENABLED */

#include "InternalScheme.template.specific.h"

#endif /* !INTERNAL_SCHEME_H */

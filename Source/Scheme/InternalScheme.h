#ifndef INTERNAL_SCHEME_H
#define INTERNAL_SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"

/**
 * Dependencies between Schemes ('|' and '<-' are used for inheritance, '!' is used for usage):
 *
 *                          Scheme1D_stype,               Scheme2D_stype,               Scheme3D_stype
 *                             |        !                    |        !                    |        !
 *                             |        !                    |        !                    |        !
 * ShemeBase           <--  Scheme1D,   !                 Scheme2D,   !                 Scheme3D    !
 *                                      !                             !                             !
 *                                      !                             !                             !
 *                          InternalScheme1D_stype,       InternalScheme2D_stype,       InternalScheme3D_stype
 *                             |                             |                             |
 *                             |                             |                             |
 * InternalSchemeBase  <--  InternalScheme1D,             InternalScheme2D,             InternalScheme3D
 */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class InternalSchemeBase;
class InternalSchemeHelper;

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class InternalSchemeBaseGPU;
class InternalSchemeHelperGPU;

/*
 * ====================================
 * ======== CPU InternalScheme ========
 * ====================================
 */
#define INTERNAL_SCHEME_BASE InternalSchemeBase
#define INTERNAL_SCHEME_BASE_CPU_FRIEND \
  template <SchemeType_t Type1, template <typename, bool> class TCoord1, LayoutType layout_type1> \
  friend class InternalSchemeBaseGPU;
#define INTERNAL_SCHEME_BASE_HELPER_CPU_FRIEND friend class InternalSchemeHelperGPU;
#define INTERNAL_SCHEME_HELPER InternalSchemeHelper
#define IGRID Grid
#define ICUDA_HOST CUDA_HOST
#define ICUDA_DEVICE

#define ALLOCATE_PARALLEL_GRIDS ICUDA_HOST virtual void allocateParallelGrids () { ALWAYS_ASSERT (0); }
#define ALLOCATE_PARALLEL_GRIDS_OVERRIDE ICUDA_HOST virtual void allocateParallelGrids () CXX11_OVERRIDE_FINAL;

#define INIT ICUDA_HOST void init (YeeGridLayout<Type, TCoord, layout_type> *layout, bool parallelLayout);
#define INIT_FROM_CPU
#define INIT_ON_GPU
#define COPY_FROM_CPU
#define COPY_TO_GPU
#define COPY_BACK_TO_CPU

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

#define ALLOCATE_GRIDS \
  ICUDA_HOST \
  void allocateGrids () \
  { \
    YeeGridLayout<Type, TCoord, layout_type> *layout = InternalSchemeBase<Type, TCoord, layout_type>::yeeLayout; \
    InternalSchemeHelper::allocateGrids<Type, TCoord, layout_type> (this, layout); \
  } \
  ICUDA_HOST \
  void allocateGridsInc () \
  { \
    YeeGridLayout<Type, TCoord, layout_type> *layout = InternalSchemeBase<Type, TCoord, layout_type>::yeeLayout; \
    InternalSchemeHelper::allocateGridsInc<Type, TCoord, layout_type> (this, layout); \
  }
#define ALLOCATE_GRIDS_GPU
#define COPY_GRIDS_GPU

#define HELPER_ALLOCATE_GRIDS \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST \
  static void allocateGrids (InternalSchemeBase<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout);
#define HELPER_ALLOCATE_GRIDS_INC \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST \
  static void allocateGridsInc (InternalSchemeBase<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout);
#define HELPER_ALLOCATE_GRIDS_FROM_CPU
#define HELPER_ALLOCATE_GRIDS_ON_GPU

#define HELPER_COPY_GRIDS_FROM_CPU
#define HELPER_COPY_GRIDS_TO_GPU

#define HELPER_ALLOCATE_PARALLEL_GRIDS \
  template <SchemeType_t Type, LayoutType layout_type> \
  ICUDA_HOST static \
  void allocateParallelGrids (ParallelYeeGridLayout<Type, layout_type> *pLayout, \
                              bool doNeedEx, bool doNeedEy, bool doNeedEz, \
                              bool doNeedHx, bool doNeedHy, bool doNeedHz, \
                              bool doNeedSigmaX, bool doNeedSigmaY, bool doNeedSigmaZ, \
                              ParallelGridCoordinate bufSize, ParallelGrid **Eps, ParallelGrid **Mu, \
                              ParallelGrid **Ex, ParallelGrid **Ey, ParallelGrid **Ez, \
                              ParallelGrid **Hx, ParallelGrid **Hy, ParallelGrid **Hz, \
                              ParallelGrid **Dx, ParallelGrid **Dy, ParallelGrid **Dz, \
                              ParallelGrid **Bx, ParallelGrid **By, ParallelGrid **Bz, \
                              ParallelGrid **D1x, ParallelGrid **D1y, ParallelGrid **D1z, \
                              ParallelGrid **B1x, ParallelGrid **B1y, ParallelGrid **B1z, \
                              ParallelGrid **SigmaX, ParallelGrid **SigmaY, ParallelGrid **SigmaZ, \
                              ParallelGrid **ExAmplitude, ParallelGrid **EyAmplitude, ParallelGrid **EzAmplitude, \
                              ParallelGrid **HxAmplitude, ParallelGrid **HyAmplitude, ParallelGrid **HzAmplitude, \
                              ParallelGrid **OmegaPE, ParallelGrid **GammaE, \
                              ParallelGrid **OmegaPM, ParallelGrid **GammaM);

#define PERFORM_FIELD_STEPS_KERNEL
#define PERFORM_PLANE_WAVE_STEPS_KERNELS
#define SHIFT_IN_TIME_KERNEL_LAUNCHES
#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCHES

#include "InternalScheme.inc.h"

/*
 * ====================================
 * ======== GPU InternalScheme ========
 * ====================================
 */
#ifdef CUDA_ENABLED

#define INTERNAL_SCHEME_BASE InternalSchemeBaseGPU
#define INTERNAL_SCHEME_BASE_CPU_FRIEND
#define INTERNAL_SCHEME_BASE_HELPER_CPU_FRIEND friend class InternalSchemeHelper;
#define INTERNAL_SCHEME_HELPER InternalSchemeHelperGPU
#define IGRID CudaGrid
#define ICUDA_HOST CUDA_HOST
#define ICUDA_DEVICE CUDA_DEVICE

#define ALLOCATE_PARALLEL_GRIDS
#define ALLOCATE_PARALLEL_GRIDS_OVERRIDE

#define INIT
#define INIT_FROM_CPU ICUDA_HOST void initFromCPU (InternalSchemeBase<Type, TCoord, layout_type> *cpuScheme, TC, TC);
#define INIT_ON_GPU ICUDA_HOST void initOnGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme);
#define COPY_FROM_CPU ICUDA_HOST void copyFromCPU (TCoord<grid_coord, true>, TCoord<grid_coord, true>);
#define COPY_TO_GPU ICUDA_HOST void copyToGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme);
#define COPY_BACK_TO_CPU ICUDA_HOST void copyBackToCPU ();

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

#define ALLOCATE_GRIDS
#define ALLOCATE_GRIDS_GPU \
  ICUDA_HOST \
  void allocateGridsFromCPU (InternalSchemeBase<Type, TCoord, layout_type> *cpuScheme, TC blockSize, TC bufSize) \
  { \
    InternalSchemeHelperGPU::allocateGridsFromCPU<Type, TCoord, layout_type> (this, cpuScheme, blockSize, bufSize); \
  } \
  ICUDA_HOST \
  void allocateGridsOnGPU () \
  { \
    InternalSchemeHelperGPU::allocateGridsOnGPU<Type, TCoord, layout_type> (this); \
  }
#define COPY_GRIDS_GPU \
  ICUDA_HOST \
  void copyGridsFromCPU (TC start, TC end) \
  { \
    InternalSchemeHelperGPU::copyGridsFromCPU<Type, TCoord, layout_type> (this, start, end); \
  } \
  ICUDA_HOST \
  void copyGridsToGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme) \
  { \
    InternalSchemeHelperGPU::copyGridsToGPU<Type, TCoord, layout_type> (this, gpuScheme); \
  } \
  ICUDA_HOST \
  void copyGridsBackToCPU () \
  { \
    InternalSchemeHelperGPU::copyGridsBackToCPU<Type, TCoord, layout_type> (this); \
  }

#define HELPER_ALLOCATE_GRIDS
#define HELPER_ALLOCATE_GRIDS_INC
#define HELPER_ALLOCATE_GRIDS_FROM_CPU \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST static void \
  allocateGridsFromCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *intScheme, \
                        InternalSchemeBase<Type, TCoord, layout_type> *cpuScheme, \
                        TCoord<grid_coord, true> blockSize, TCoord<grid_coord, true> bufSize);
#define HELPER_ALLOCATE_GRIDS_ON_GPU \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>  \
  ICUDA_HOST static void allocateGridsOnGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme);

#define HELPER_COPY_GRIDS_FROM_CPU \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST static void \
  copyGridsFromCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme, \
                    TCoord<grid_coord, true> start, TCoord<grid_coord, true> end); \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST static void \
  copyGridsBackToCPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme);
#define HELPER_COPY_GRIDS_TO_GPU \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  ICUDA_HOST static void \
  copyGridsToGPU (InternalSchemeBaseGPU<Type, TCoord, layout_type> *intScheme, \
                  InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme);

#define HELPER_ALLOCATE_PARALLEL_GRIDS

#define SETUP_BLOCKS_AND_THREADS \
  GridCoordinate3D diff3D = expandTo3D (diff, ct1, ct2, ct3); \
  ASSERT (diff3D.get1 () % SOLVER_SETTINGS.getNumCudaThreadsX () == 0); \
  ASSERT (diff3D.get2 () % SOLVER_SETTINGS.getNumCudaThreadsY () == 0); \
  ASSERT (diff3D.get3 () % SOLVER_SETTINGS.getNumCudaThreadsZ () == 0); \
  dim3 blocks (diff3D.get1 () == 0 ? 1 : diff3D.get1 () / SOLVER_SETTINGS.getNumCudaThreadsX (), \
               diff3D.get2 () == 0 ? 1 : diff3D.get2 () / SOLVER_SETTINGS.getNumCudaThreadsY (), \
               diff3D.get3 () == 0 ? 1 : diff3D.get3 () / SOLVER_SETTINGS.getNumCudaThreadsZ ()); \
  dim3 threads (diff3D.get1 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsX (), \
                diff3D.get2 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsY (), \
                diff3D.get3 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsZ ()); \

#define PERFORM_FIELD_STEPS_KERNEL \
  template <uint8_t grid_type> \
  CUDA_HOST \
  void performFieldStepsKernelLaunch (time_step t, TC Start, TC End, CoordinateType ct1, CoordinateType ct2, CoordinateType ct3) \
  { \
    TC diff = End - Start; \
    SETUP_BLOCKS_AND_THREADS; \
    TC one = TC_COORD (1, 1, 1, ct1, ct2, ct3); \
    InternalSchemeKernelHelpers::performFieldStepsKernel<Type, TCoord, layout_type, grid_type> <<< blocks, threads >>> (this, t, Start, one, ct1, ct2, ct3); \
    cudaCheckError (); \
  }

#define PERFORM_PLANE_WAVE_STEPS_KERNELS \
  CUDA_HOST \
  void performPlaneWaveEStepsKernelLaunch (time_step t, GridCoordinate1D Start, GridCoordinate1D End) \
  { \
    GridCoordinate1D diff = End - Start; \
    int thrds = SOLVER_SETTINGS.getNumCudaThreadsX () \
                  * SOLVER_SETTINGS.getNumCudaThreadsY () \
                  * SOLVER_SETTINGS.getNumCudaThreadsZ (); \
    ASSERT (diff.get1 () % thrds == 0); \
    dim3 blocks (diff.get1 () / thrds, 1, 1); \
    dim3 threads (thrds, 1, 1); \
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X); \
    InternalSchemeKernelHelpers::performPlaneWaveEStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (this, t, Start, one); \
    cudaCheckError (); \
  } \
  CUDA_HOST \
  void performPlaneWaveHStepsKernelLaunch (time_step t, GridCoordinate1D Start, GridCoordinate1D End) \
  { \
    GridCoordinate1D diff = End - Start; \
    int thrds = SOLVER_SETTINGS.getNumCudaThreadsX () \
                  * SOLVER_SETTINGS.getNumCudaThreadsY () \
                  * SOLVER_SETTINGS.getNumCudaThreadsZ (); \
    ASSERT (diff.get1 () % thrds == 0); \
    dim3 blocks (diff.get1 () / thrds, 1, 1); \
    dim3 threads (thrds, 1, 1); \
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X); \
    InternalSchemeKernelHelpers::performPlaneWaveHStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (this, t, Start, one); \
    cudaCheckError (); \
  }

#define SHIFT_IN_TIME_KERNEL(NAME) \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void shiftInTimeKernel ## NAME (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme, \
                                  TCoord<grid_coord, true> Start, TCoord<grid_coord, true> CoordPerKernel, \
                                  CoordinateType ct1, CoordinateType ct2, CoordinateType ct3) \
  { \
    TCoord<grid_coord, true> posStart = TCoord<grid_coord, true>::initAxesCoordinate ((blockIdx.x * blockDim.x) + threadIdx.x, \
                                          (blockIdx.y * blockDim.y) + threadIdx.y, \
                                          (blockIdx.z * blockDim.z) + threadIdx.z, \
                                          ct1, ct2, ct3); \
    posStart = posStart * CoordPerKernel + Start; \
    TCoord<grid_coord, true> posEnd = posStart + CoordPerKernel; \
    gpuScheme-> get ## NAME () -> shiftInTime (posStart, posEnd); \
  } \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void nextTimeStepKernel ## NAME (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme) \
  { \
    gpuScheme-> get ## NAME () -> nextTimeStep (); \
  }

#define SHIFT_IN_TIME_KERNEL_LAUNCH(NAME) \
  CUDA_HOST \
  void shiftInTimeKernelLaunch ## NAME (TC Start, TC End, CoordinateType ct1, CoordinateType ct2, CoordinateType ct3) \
  { \
    TC diff = End - Start; \
    SETUP_BLOCKS_AND_THREADS; \
    TC one = TC_COORD (1, 1, 1, ct1, ct2, ct3); \
    InternalSchemeKernelHelpers::shiftInTimeKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (this, Start, one, ct1, ct2, ct3); \
    cudaCheckError (); \
  } \
  CUDA_HOST \
  void nextTimeStepKernelLaunch ## NAME () \
  { \
    dim3 blocks (1, 1, 1); \
    dim3 threads (1, 1, 1); \
    InternalSchemeKernelHelpers::nextTimeStepKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (this); \
    cudaCheckError (); \
  }

#define SHIFT_IN_TIME_KERNELS \
  SHIFT_IN_TIME_KERNEL(Ex) \
  SHIFT_IN_TIME_KERNEL(Ey) \
  SHIFT_IN_TIME_KERNEL(Ez) \
  SHIFT_IN_TIME_KERNEL(Hx) \
  SHIFT_IN_TIME_KERNEL(Hy) \
  SHIFT_IN_TIME_KERNEL(Hz) \
  SHIFT_IN_TIME_KERNEL(Dx) \
  SHIFT_IN_TIME_KERNEL(Dy) \
  SHIFT_IN_TIME_KERNEL(Dz) \
  SHIFT_IN_TIME_KERNEL(Bx) \
  SHIFT_IN_TIME_KERNEL(By) \
  SHIFT_IN_TIME_KERNEL(Bz) \
  SHIFT_IN_TIME_KERNEL(D1x) \
  SHIFT_IN_TIME_KERNEL(D1y) \
  SHIFT_IN_TIME_KERNEL(D1z) \
  SHIFT_IN_TIME_KERNEL(B1x) \
  SHIFT_IN_TIME_KERNEL(B1y) \
  SHIFT_IN_TIME_KERNEL(B1z) \

#define SHIFT_IN_TIME_KERNEL_LAUNCHES \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Ex) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Ey) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Ez) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hx) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hy) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hz) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dx) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dy) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dz) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Bx) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(By) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(Bz) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1x) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1y) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1z) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1x) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1y) \
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1z) \

#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL(NAME) \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void shiftInTimePlaneWaveKernel ## NAME (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme, \
                                           TCoord<grid_coord, true> Start, TCoord<grid_coord, true> CoordPerKernel) \
  { \
    GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x, \
                                 CoordinateType::X); \
    posStart = posStart * CoordPerKernel + Start; \
    TCoord<grid_coord, true> posEnd = posStart + CoordPerKernel; \
    gpuScheme-> get ## NAME () -> shiftInTime (posStart, posEnd); \
  } \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void nextTimeStepPlaneWaveKernel ## NAME (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme) \
  { \
    gpuScheme-> get ## NAME () -> nextTimeStep (); \
  } \

#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(NAME) \
  CUDA_HOST \
  void shiftInTimePlaneWaveKernelLaunch ## NAME (TC Start, TC End) \
  { \
    GridCoordinate1D diff = End - Start; \
    int thrds = SOLVER_SETTINGS.getNumCudaThreadsX () \
                  * SOLVER_SETTINGS.getNumCudaThreadsY () \
                  * SOLVER_SETTINGS.getNumCudaThreadsZ (); \
    ASSERT (diff.get1 () % thrds == 0); \
    dim3 blocks (diff.get1 () / thrds, 1, 1); \
    dim3 threads (thrds, 1, 1); \
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X); \
    InternalSchemeKernelHelpers::shiftInTimePlaneWaveKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (this, Start, one); \
    cudaCheckError (); \
  } \
  CUDA_HOST \
  void nextTimeStepPlaneWaveKernelLaunch ## NAME () \
  { \
    dim3 blocks (1, 1, 1); \
    dim3 threads (1, 1, 1); \
    InternalSchemeKernelHelpers::nextTimeStepPlaneWaveKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (this); \
    cudaCheckError (); \
  }

#define SHIFT_IN_TIME_PLANE_WAVE_KERNELS \
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL(EInc) \
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL(HInc)

#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCHES \
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(EInc) \
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(HInc)

  namespace InternalSchemeKernelHelpers
  {
    template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
    __global__
    void performFieldStepsKernel (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme,
                                  time_step t, TCoord<grid_coord, true> Start, TCoord<grid_coord, true> CoordPerKernel,
                                  CoordinateType ct1, CoordinateType ct2, CoordinateType ct3)
    {
      TCoord<grid_coord, true> posStart = TCoord<grid_coord, true>::initAxesCoordinate ((blockIdx.x * blockDim.x) + threadIdx.x,
                                            (blockIdx.y * blockDim.y) + threadIdx.y,
                                            (blockIdx.z * blockDim.z) + threadIdx.z,
                                            ct1, ct2, ct3);
      posStart = posStart * CoordPerKernel + Start;
      TCoord<grid_coord, true> posEnd = posStart + CoordPerKernel;
      gpuScheme->performFieldSteps<grid_type> (t, posStart, posEnd);
    }

    template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
    __global__
    void performPlaneWaveEStepsKernel (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme,
                                       time_step t, GridCoordinate1D Start, GridCoordinate1D CoordPerKernel)
    {
      GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                   CoordinateType::X);
      posStart = posStart * CoordPerKernel + Start;
      GridCoordinate1D posEnd = posStart + CoordPerKernel;
      gpuScheme->performPlaneWaveESteps (t, posStart, posEnd);
    }
      template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
    __global__
    void performPlaneWaveHStepsKernel (InternalSchemeBaseGPU<Type, TCoord, layout_type> *gpuScheme,
                                       time_step t, GridCoordinate1D Start, GridCoordinate1D CoordPerKernel)
    {
      GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                   CoordinateType::X);
      posStart = posStart * CoordPerKernel + Start;
      GridCoordinate1D posEnd = posStart + CoordPerKernel;
      gpuScheme->performPlaneWaveHSteps (t, posStart, posEnd);
    }

    SHIFT_IN_TIME_KERNELS
    SHIFT_IN_TIME_PLANE_WAVE_KERNELS
  };

#include "InternalScheme.inc.h"

#endif /* CUDA_ENABLED */

#include "InternalScheme.template.specific.h"

#endif /* !INTERNAL_SCHEME_H */

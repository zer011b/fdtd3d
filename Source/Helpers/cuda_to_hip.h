/*
 * CUDA to HIP compatibility header for fdtd3d.
 *
 * This header aliases CUDA runtime API symbols to their HIP equivalents
 * when building for AMD GPUs. On NVIDIA, it is a no-op that includes
 * the CUDA runtime. The rest of the codebase uses CUDA spelling.
 */

#ifndef CUDA_TO_HIP_H
#define CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

/* Runtime API symbols */
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemset              hipMemset
#define cudaMemcpyToSymbol      hipMemcpyToSymbol
#define cudaMemcpyFromSymbol    hipMemcpyFromSymbol

/* Memory copy kinds */
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

/* Error handling */
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString

/* Device management */
#define cudaSetDevice           hipSetDevice
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp          hipDeviceProp_t

/* Compiler detection: HIP uses __HIPCC__ instead of __CUDACC__ */
#if defined(__HIPCC__) && !defined(__CUDACC__)
#define __CUDACC__
#endif

/* Device compilation: HIP uses __HIP_DEVICE_COMPILE__ instead of __CUDA_ARCH__ */
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 600  /* Pretend sm_60+ so native atomicAdd(double*) is used */
#endif

#else /* CUDA path */

#include <cuda_runtime.h>

#endif /* USE_HIP */

#endif /* CUDA_TO_HIP_H */

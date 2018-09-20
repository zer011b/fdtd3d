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

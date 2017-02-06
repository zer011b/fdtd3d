#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H

enum CudaExitStatus
{
  CUDA_OK,
  CUDA_ERROR
};

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

#endif /* CUDA_DEFINES_H */

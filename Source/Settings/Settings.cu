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
  Settings *d_tmp;
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

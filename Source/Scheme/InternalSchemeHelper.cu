#define CUDA_SOURCES

#include "InternalSchemeHelper.h"

#ifdef CUDA_ENABLED

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalSchemeHelper::allocateGridsOnGPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *gpuScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Eps, sizeof(TGrid<TC>)));
  cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Mu, sizeof(TGrid<TC>)));

  if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ex, sizeof(TGrid<TC>))); }
  if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ey, sizeof(TGrid<TC>))); }
  if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Ez, sizeof(TGrid<TC>))); }
  if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hx, sizeof(TGrid<TC>))); }
  if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hy, sizeof(TGrid<TC>))); }
  if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Hz, sizeof(TGrid<TC>))); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dx, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dy, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Dz, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Bx, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->By, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->Bz, sizeof(TGrid<TC>))); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1x, sizeof(TGrid<TC>))); }
      if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1y, sizeof(TGrid<TC>))); }
      if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->D1z, sizeof(TGrid<TC>))); }
      if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1x, sizeof(TGrid<TC>))); }
      if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1y, sizeof(TGrid<TC>))); }
      if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->B1z, sizeof(TGrid<TC>))); }
    }

    if (gpuScheme->doNeedSigmaX) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaX, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedSigmaY) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaY, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedSigmaZ) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->SigmaZ, sizeof(TGrid<TC>))); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->ExAmplitude, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EyAmplitude, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EzAmplitude, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HxAmplitude, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HyAmplitude, sizeof(TGrid<TC>))); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HzAmplitude, sizeof(TGrid<TC>))); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->OmegaPE, sizeof(TGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->GammaE, sizeof(TGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->OmegaPM, sizeof(TGrid<TC>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->GammaM, sizeof(TGrid<TC>)));
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EInc, sizeof(TGrid<GridCoordinate1D>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HInc, sizeof(TGrid<GridCoordinate1D>)));
  }
}

#endif /* CUDA_ENABLED */

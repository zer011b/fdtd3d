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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalSchemeHelper::copyGridsToGPU (InternalSchemeBase<Type, TCoord, layout_type, TGrid> *intScheme,
                                      InternalSchemeBase<Type, TCoord, layout_type, TGrid> *gpuScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Eps, intScheme->Eps, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Mu, intScheme->Mu, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));

  if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ex, intScheme->Ex, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ey, intScheme->Ey, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Ez, intScheme->Ez, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hx, intScheme->Hx, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hy, intScheme->Hy, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Hz, intScheme->Hz, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dx, intScheme->Dx, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dy, intScheme->Dy, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Dz, intScheme->Dz, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Bx, intScheme->Bx, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->By, intScheme->By, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->Bz, intScheme->Bz, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1x, intScheme->D1x, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1y, intScheme->D1y, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->D1z, intScheme->D1z, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1x, intScheme->B1x, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1y, intScheme->B1y, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
      if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->B1z, intScheme->B1z, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    }

    if (gpuScheme->doNeedSigmaX) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaX, intScheme->SigmaX, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedSigmaY) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaY, intScheme->SigmaY, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedSigmaZ) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->SigmaZ, intScheme->SigmaZ, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  }

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    if (gpuScheme->doNeedEx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->ExAmplitude, intScheme->ExAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EyAmplitude, intScheme->EyAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedEz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EzAmplitude, intScheme->EzAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHx) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HxAmplitude, intScheme->HxAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHy) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HyAmplitude, intScheme->HyAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
    if (gpuScheme->doNeedHz) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HzAmplitude, intScheme->HzAmplitude, sizeof(TGrid<TC>), cudaMemcpyHostToDevice)); }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->OmegaPE, intScheme->OmegaPE, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->GammaE, intScheme->GammaE, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->OmegaPM, intScheme->OmegaPM, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->GammaM, intScheme->GammaM, sizeof(TGrid<TC>), cudaMemcpyHostToDevice));
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EInc, intScheme->EInc, sizeof(TGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HInc, intScheme->HInc, sizeof(TGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
  }
}

#endif /* CUDA_ENABLED */

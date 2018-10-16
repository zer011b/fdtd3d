#define CUDA_SOURCES

#ifdef CUDA_ENABLED

#include "InternalScheme.h"

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::initFromCPU (InternalSchemeBase<Type, TCoord, layout_type, Grid> *cpuScheme,
                                                                  TC blockSize,
                                                                  TC bufSize)
{
  ASSERT (cpuScheme->isInitialized);

  cudaCheckErrorCmd (cudaMalloc ((void **) &yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>)));
  cudaCheckErrorCmd (cudaMemcpy (yeeLayout, cpuScheme->yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));

  useParallel = false;

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = cpuScheme->yeeLayout->getEzSize () - leftNTFF + one;
  }

  allocateGridsFromCPU (cpuScheme, blockSize, bufSize);

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, template <typename> class TGrid>
CUDA_HOST
void
InternalSchemeBase<Type, TCoord, layout_type, TGrid>::initOnGPU ()
{
  ASSERT (isInitialized);

  allocateGridsOnGPU ();
}

#include "InternalScheme.cpp"

template class InternalScheme1D_ExHy_CudaGrid<E_CENTERED>;
template class InternalScheme1D_ExHz_CudaGrid<E_CENTERED>;
template class InternalScheme1D_EyHx_CudaGrid<E_CENTERED>;
template class InternalScheme1D_EyHz_CudaGrid<E_CENTERED>;
template class InternalScheme1D_EzHx_CudaGrid<E_CENTERED>;
template class InternalScheme1D_EzHy_CudaGrid<E_CENTERED>;

template class InternalScheme2D_TEx_CudaGrid<E_CENTERED>;
template class InternalScheme2D_TEy_CudaGrid<E_CENTERED>;
template class InternalScheme2D_TEz_CudaGrid<E_CENTERED>;
template class InternalScheme2D_TMx_CudaGrid<E_CENTERED>;
template class InternalScheme2D_TMy_CudaGrid<E_CENTERED>;
template class InternalScheme2D_TMz_CudaGrid<E_CENTERED>;

template class InternalScheme3D_3D_CudaGrid<E_CENTERED>;

#endif /* CUDA_ENABLED */

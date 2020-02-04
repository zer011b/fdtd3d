/*
 * Unit test for InternalScheme on CPU
 */

#include <iostream>

#include "InternalScheme.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

#define SIZE 20
#define PML_SIZE 5
#define TFSF_SIZE 7

#define LAMBDA 0.2
#define DX 0.002

#define ACCURACY 0.000018

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void test (InternalScheme<Type, TCoord, layout_type> *intScheme,
           TCoord<grid_coord, true> overallSize,
           TCoord<grid_coord, true> pmlSize,
           TCoord<grid_coord, true> tfsfSizeLeft,
           TCoord<grid_coord, true> tfsfSizeRight,
           CoordinateType ct1,
           CoordinateType ct2,
           CoordinateType ct3)
{
  intScheme->getEps ()->initialize (FieldValueHelpers::getFieldValueRealOnly (1.0));
  intScheme->getMu ()->initialize (FieldValueHelpers::getFieldValueRealOnly (1.0));

  if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
  {
    if (intScheme->getDoNeedEx ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getEx ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getEx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getExStartDiff () && pos < intScheme->getEx ()->getSize () - intScheme->getYeeLayout ()->getExEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEx ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EX, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEx ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getCbEx ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getEy ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getEy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getEyStartDiff () && pos < intScheme->getEy ()->getSize () - intScheme->getYeeLayout ()->getEyEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEy ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EY, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEy ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getCbEy ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getEz ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getEz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getEzStartDiff () && pos < intScheme->getEz ()->getSize () - intScheme->getYeeLayout ()->getEzEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getEz ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::EZ, intScheme->getEps (), GridType::EPS);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * DX);

        intScheme->getCaEz ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getCbEz ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getHx ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getHx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getHxStartDiff () && pos < intScheme->getHx ()->getSize () - intScheme->getYeeLayout ()->getHxEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHx ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HX, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHx ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getDbHx ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getHy ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getHy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getHyStartDiff () && pos < intScheme->getHy ()->getSize () - intScheme->getYeeLayout ()->getHyEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHy ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HY, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHy ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getDbHy ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter = intScheme->getHz ()->begin ();
      typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = intScheme->getHz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TCoord<grid_coord, true> pos = iter.getPos ();
        if (!(pos >= intScheme->getYeeLayout ()->getHzStartDiff () && pos < intScheme->getHz ()->getSize () - intScheme->getYeeLayout ()->getHzEndDiff ()))
        {
          continue;
        }

        TCoord<grid_coord, true> posAbs = intScheme->getHz ()->getTotalPosition (pos);

        FPValue material = intScheme->getMaterial (posAbs, GridType::HZ, intScheme->getMu (), GridType::MU);

        FPValue ca = 1.0;
        FPValue cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * DX);

        intScheme->getDaHz ()->setFieldValue (FIELDVALUE (ca, 0), pos, 0);
        intScheme->getDbHz ()->setFieldValue (FIELDVALUE (cb, 0), pos, 0);
      }
    }
  }

#ifdef CUDA_ENABLED
  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntScheme = NULLPTR;
  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntSchemeOnGPU = NULLPTR;
  InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuIntSchemeOnGPU = NULLPTR;

  if (SOLVER_SETTINGS.getDoUseCuda ())
  {
    int cudaBuf = 0;

    gpuIntScheme = new InternalSchemeGPU<Type, TCoord, layout_type> ();
    gpuIntSchemeOnGPU = new InternalSchemeGPU<Type, TCoord, layout_type> ();

    TCoord<grid_coord, true> buf = TCoord<grid_coord, true>::initAxesCoordinate (cudaBuf, cudaBuf, cudaBuf, ct1, ct2, ct3);
    TCoord<grid_coord, true> zero = TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

    gpuIntScheme->initFromCPU (intScheme, intScheme->getYeeLayout ()->getSize (), buf);
    gpuIntSchemeOnGPU->initOnGPU (gpuIntScheme);
    cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>)));

    gpuIntScheme->copyFromCPU (zero, intScheme->getYeeLayout ()->getSize ());
    gpuIntSchemeOnGPU->copyToGPU (gpuIntScheme);
    cudaCheckErrorCmd (cudaMemcpy (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));
  }
#endif /* CUDA_ENABLED */

  for (time_step t = 0; t < SOLVER_SETTINGS.getNumTimeSteps (); ++t)
  {
    DPRINTF (LOG_LEVEL_NONE, "calculating time step %d\n", t);

    FPValue timestep = 0;

    TCoord<grid_coord, true> ExStart, ExEnd;
    TCoord<grid_coord, true> EyStart, EyEnd;
    TCoord<grid_coord, true> EzStart, EzEnd;
    TCoord<grid_coord, true> HxStart, HxEnd;
    TCoord<grid_coord, true> HyStart, HyEnd;
    TCoord<grid_coord, true> HzStart, HzEnd;

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ())
    {
      ExStart = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getExStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      ExEnd = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getExEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      EyStart = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getEyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      EyEnd = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getEyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      EzStart = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getEzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      EzEnd = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getEzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HxStart = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getHxStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HxEnd = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getHxEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HyStart = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getHyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HyEnd = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getHyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HzStart = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationStart (gpuIntScheme->getYeeLayout ()->getHzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HzEnd = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationEnd (gpuIntScheme->getYeeLayout ()->getHzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    }
    else
#endif /* CUDA_ENABLED */
    {
      ExStart = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationStart (intScheme->getYeeLayout ()->getExStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      ExEnd = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationEnd (intScheme->getYeeLayout ()->getExEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      EyStart = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationStart (intScheme->getYeeLayout ()->getEyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      EyEnd = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationEnd (intScheme->getYeeLayout ()->getEyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      EzStart = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationStart (intScheme->getYeeLayout ()->getEzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      EzEnd = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationEnd (intScheme->getYeeLayout ()->getEzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HxStart = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationStart (intScheme->getYeeLayout ()->getHxStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HxEnd = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationEnd (intScheme->getYeeLayout ()->getHxEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HyStart = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationStart (intScheme->getYeeLayout ()->getHyStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HyEnd = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationEnd (intScheme->getYeeLayout ()->getHyEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);

      HzStart = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationStart (intScheme->getYeeLayout ()->getHzStartDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
      HzEnd = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationEnd (intScheme->getYeeLayout ()->getHzEndDiff ()) : TCoord<grid_coord, true>::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        gpuIntSchemeOnGPU->performPlaneWaveEStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getEInc ()->getSize ());
        gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchEInc (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEInc ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        intScheme->performPlaneWaveESteps (t, zero1D, intScheme->getEInc ()->getSize ());
        intScheme->getEInc ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedEx ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::EX) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (ExStart, ExEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getCaEx ();
          d_Cb = gpuIntSchemeOnGPU->getCbEx ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::EX)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getEx (),
                                                                                gpuIntScheme->getDoNeedHz () ? gpuIntSchemeOnGPU->getHz () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedHy () ? gpuIntSchemeOnGPU->getHy () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::EX, gpuIntSchemeOnGPU->getEps (), GridType::EPS,
                                                                                PhysicsConst::Eps0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEx ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getEx ()->getTotalPosition (pos);

          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EX), true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEx (), coordFP,
                                                               intScheme->getDoNeedHz () ? intScheme->getHz () : NULLPTR,
                                                               intScheme->getDoNeedHy () ? intScheme->getHy () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getCaEx (), intScheme->getCbEx (),
                                                               false,
                                                               GridType::EX, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EX), false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEx (), coordFP,
                                                               intScheme->getDoNeedHz () ? intScheme->getHz () : NULLPTR,
                                                               intScheme->getDoNeedHy () ? intScheme->getHy () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::EX, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }

        intScheme->getEx ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::EY) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (EyStart, EyEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getCaEy ();
          d_Cb = gpuIntSchemeOnGPU->getCbEy ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::EY)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getEy (),
                                                                                gpuIntScheme->getDoNeedHx () ? gpuIntSchemeOnGPU->getHx () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedHz () ? gpuIntSchemeOnGPU->getHz () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::EY, gpuIntSchemeOnGPU->getEps (), GridType::EPS,
                                                                                PhysicsConst::Eps0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEy ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getEy ()->getTotalPosition (pos);
          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EY) , true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEy (), coordFP,
                                                               intScheme->getDoNeedHx () ? intScheme->getHx () : NULLPTR,
                                                               intScheme->getDoNeedHz () ? intScheme->getHz () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getCaEy (), intScheme->getCbEy (),
                                                               false,
                                                               GridType::EY, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EY) , false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEy (), coordFP,
                                                               intScheme->getDoNeedHx () ? intScheme->getHx () : NULLPTR,
                                                               intScheme->getDoNeedHz () ? intScheme->getHz () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::EY, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }

        intScheme->getEy ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::EZ) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (EzStart, EzEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getCaEz ();
          d_Cb = gpuIntSchemeOnGPU->getCbEz ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::EZ)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getEz (),
                                                                                gpuIntScheme->getDoNeedHy () ? gpuIntSchemeOnGPU->getHy () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedHx () ? gpuIntSchemeOnGPU->getHx () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::EZ, gpuIntSchemeOnGPU->getEps (), GridType::EPS,
                                                                                PhysicsConst::Eps0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEz ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getEz ()->getTotalPosition (pos);
          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EZ) , true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEz (), coordFP,
                                                               intScheme->getDoNeedHy () ? intScheme->getHy () : NULLPTR,
                                                               intScheme->getDoNeedHx () ? intScheme->getHx () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getCaEz (), intScheme->getCbEz (),
                                                               false,
                                                               GridType::EZ, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::EZ) , false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getEz (), coordFP,
                                                               intScheme->getDoNeedHy () ? intScheme->getHy () : NULLPTR,
                                                               intScheme->getDoNeedHx () ? intScheme->getHx () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::EZ, intScheme->getEps (), GridType::EPS,
                                                               PhysicsConst::Eps0);
          }
        }

        intScheme->getEz ()->shiftInTime ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        gpuIntSchemeOnGPU->performPlaneWaveHStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getHInc ()->getSize ());
        gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchHInc (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHInc ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        intScheme->performPlaneWaveHSteps (t, zero1D, intScheme->getHInc ()->getSize ());
        intScheme->getHInc ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::HX) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HxStart, HxEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getDaHx ();
          d_Cb = gpuIntSchemeOnGPU->getDbHx ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::HX)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getHx (),
                                                                                gpuIntScheme->getDoNeedEy () ? gpuIntSchemeOnGPU->getEy () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedEz () ? gpuIntSchemeOnGPU->getEz () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::HX, gpuIntSchemeOnGPU->getMu (), GridType::MU,
                                                                                PhysicsConst::Mu0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHx ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getHx ()->getTotalPosition (pos);
          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HX), true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHx (), coordFP,
                                                               intScheme->getDoNeedEy () ? intScheme->getEy () : NULLPTR,
                                                               intScheme->getDoNeedEz () ? intScheme->getEz () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getDaHx (), intScheme->getDbHx (),
                                                               false,
                                                               GridType::HX, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HX), false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHx (), coordFP,
                                                               intScheme->getDoNeedEy () ? intScheme->getEy () : NULLPTR,
                                                               intScheme->getDoNeedEz () ? intScheme->getEz () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::HX, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }

        intScheme->getHx ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::HY) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HyStart, HyEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getDaHy ();
          d_Cb = gpuIntSchemeOnGPU->getDbHy ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::HY)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getHy (),
                                                                                gpuIntScheme->getDoNeedEz () ? gpuIntSchemeOnGPU->getEz () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedEx () ? gpuIntSchemeOnGPU->getEx () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::HY, gpuIntSchemeOnGPU->getMu (), GridType::MU,
                                                                                PhysicsConst::Mu0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHy ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getHy ()->getTotalPosition (pos);
          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HY), true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHy (), coordFP,
                                                               intScheme->getDoNeedEz () ? intScheme->getEz () : NULLPTR,
                                                               intScheme->getDoNeedEx () ? intScheme->getEx () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getDaHy (), intScheme->getDbHy (),
                                                               false,
                                                               GridType::HY, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HY), false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHy (), coordFP,
                                                               intScheme->getDoNeedEz () ? intScheme->getEz () : NULLPTR,
                                                               intScheme->getDoNeedEx () ? intScheme->getEx () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::HY, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }

        intScheme->getHy ()->shiftInTime ();
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      TCoord<grid_coord, false> diff11;
      TCoord<grid_coord, false> diff12;
      TCoord<grid_coord, false> diff21;
      TCoord<grid_coord, false> diff22;

      TCoord<FPValue, true> coordFP;

      intScheme->template calculateFieldStepInitDiff< static_cast<uint8_t> (GridType::HZ) > (&diff11, &diff12, &diff21, &diff22);

      GridCoordinate3D start3D;
      GridCoordinate3D end3D;

      expandTo3DStartEnd (HzStart, HzEnd, start3D, end3D, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ())
      {
        CudaGrid< TCoord<grid_coord, true> > *d_Ca = NULLPTR;
        CudaGrid< TCoord<grid_coord, true> > *d_Cb = NULLPTR;
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          d_Ca = gpuIntSchemeOnGPU->getDaHz ();
          d_Cb = gpuIntSchemeOnGPU->getDbHz ();
        }

        gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <static_cast<uint8_t> (GridType::HZ)> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                                timestep, diff11, diff12, diff21, diff22,
                                                                                gpuIntSchemeOnGPU->getHz (),
                                                                                gpuIntScheme->getDoNeedEx () ? gpuIntSchemeOnGPU->getEx () : NULLPTR,
                                                                                gpuIntScheme->getDoNeedEy () ? gpuIntSchemeOnGPU->getEy () : NULLPTR,
                                                                                NULLPTR,
                                                                                d_Ca,
                                                                                d_Cb,
                                                                                false,
                                                                                GridType::HZ, gpuIntSchemeOnGPU->getMu (), GridType::MU,
                                                                                PhysicsConst::Mu0,
                                                                                SOLVER_SETTINGS.getDoUseCaCbGrids ());

        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHz ()->shiftInTime ();
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter (start, start, end);
        typename VectorFieldValues< TCoord<grid_coord, true> >::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TCoord<grid_coord, true> pos = iter.getPos ();
          TCoord<grid_coord, true> posAbs = intScheme->getHz ()->getTotalPosition (pos);
          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HZ), true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHz (), coordFP,
                                                               intScheme->getDoNeedEx () ? intScheme->getEx () : NULLPTR,
                                                               intScheme->getDoNeedEy () ? intScheme->getEy () : NULLPTR,
                                                               NULLPTR,
                                                               intScheme->getDaHz (), intScheme->getDbHz (),
                                                               false,
                                                               GridType::HZ, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
          else
          {
            intScheme->template calculateFieldStepIteration< static_cast<uint8_t> (GridType::HZ), false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               intScheme->getHz (), coordFP,
                                                               intScheme->getDoNeedEx () ? intScheme->getEx () : NULLPTR,
                                                               intScheme->getDoNeedEy () ? intScheme->getEy () : NULLPTR,
                                                               NULLPTR,
                                                               NULLPTR, NULLPTR,
                                                               false,
                                                               GridType::HZ, intScheme->getMu (), GridType::MU,
                                                               PhysicsConst::Mu0);
          }
        }

        intScheme->getHz ()->shiftInTime ();
      }
    }
  }

#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ())
  {
    gpuIntScheme->copyBackToCPU (SOLVER_SETTINGS.getNumTimeSteps (), true);

    /*
     * Free memory
     */
    if (d_gpuIntSchemeOnGPU)
    {
      cudaCheckErrorCmd (cudaFree (d_gpuIntSchemeOnGPU));
    }

    if (gpuIntSchemeOnGPU)
    {
      gpuIntSchemeOnGPU->uninitOnGPU ();
    }
    if (gpuIntScheme)
    {
      gpuIntScheme->uninitFromCPU ();
    }

    delete gpuIntSchemeOnGPU;
    delete gpuIntScheme;
  }
#endif /* CUDA_ENABLED */
}

#if defined (MODE_DIM1)

void test1D (Grid<GridCoordinate1D> *E,
             GridCoordinateFP1D diff,
             CoordinateType ct1,
             Grid<GridCoordinate1D> *E_cmp)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                            , ct1
#endif /* DEBUG_INFO */
                            );
      GridCoordinateFP1D posFP (i
#ifdef DEBUG_INFO
                                , ct1
#endif /* DEBUG_INFO */
                                );
      posFP = posFP + diff;

      FieldValue val = *E->getFieldValue (pos, 1);

      if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE)
      {
        ASSERT (SQR (val.abs () - FPValue (1)) < ACCURACY);
      }
      else
      {
        ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
      }

#ifdef DOUBLE_VALUES
      /*
       * For double values
       */
      if (SIZE == 20)
      {
        FieldValue cmp = *E_cmp->getFieldValue (pos, 0);
        ASSERT (IS_FP_EXACT (val.real (), cmp.real ()));
        ASSERT (IS_FP_EXACT (val.imag (), cmp.imag ()));
      }
#endif /* DOUBLE_VALUES */
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)

void test2D (Grid<GridCoordinate2D> *E,
             GridCoordinateFP2D diff,
             CoordinateType ct1,
             CoordinateType ct2)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      for (grid_coord j = 0; j < SIZE; ++j)
      {
        GridCoordinate2D pos (i, j
#ifdef DEBUG_INFO
                              , ct1, ct2
#endif /* DEBUG_INFO */
                              );
        GridCoordinateFP2D posFP (i, j
#ifdef DEBUG_INFO
                                  , ct1, ct2
#endif /* DEBUG_INFO */
                                  );
        posFP = posFP + diff;

        FieldValue val = *E->getFieldValue (pos, 1);

        if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
            && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE)
        {
          ASSERT (SQR (val.abs () - FPValue (1)) < ACCURACY);
        }
        else
        {
          ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
        }
      }
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)

void test3D (Grid<GridCoordinate3D> *E,
             GridCoordinateFP3D diff,
             CoordinateType ct1,
             CoordinateType ct2,
             CoordinateType ct3)
{
#ifdef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    for (grid_coord i = 0; i < SIZE; ++i)
    {
      for (grid_coord j = 0; j < SIZE; ++j)
      {
        for (grid_coord k = 0; k < SIZE; ++k)
        {
          GridCoordinate3D pos (i, j, k
#ifdef DEBUG_INFO
                                , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                );
          GridCoordinateFP3D posFP (i, j, k
#ifdef DEBUG_INFO
                                    , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                    );
          posFP = posFP + diff;

          FieldValue val = *E->getFieldValue (pos, 1);

          if (posFP.get1 () >= TFSF_SIZE && posFP.get1 () <= SIZE - TFSF_SIZE
              && posFP.get2 () >= TFSF_SIZE && posFP.get2 () <= SIZE - TFSF_SIZE
              && posFP.get3 () >= TFSF_SIZE && posFP.get3 () <= SIZE - TFSF_SIZE)
          {
            ASSERT (SQR (val.abs () - FPValue (1)) < ACCURACY);
          }
          else
          {
            ASSERT (IS_FP_EXACT (val.abs (), FPValue (0)));
          }
        }
      }
    }
  }
#endif /* COMPLEX_FIELD_VALUES */
}

#endif /* MODE_DIM3 */

#if defined (MODE_EX_HY)
template<LayoutType layout_type>
void test1D_ExHy ()
{
  CoordinateType ct1 = CoordinateType::Z;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 0;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ex_cmp (overallSize, 1);
  cmp = &Ex_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0.21811863445648658, -0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.27908260941002716, -0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.3386998783352157, -0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.39727542433754848, -0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.45394742244500197, -0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.50918770413007142, -0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0.18730018750160604, -0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.24860062193325649, -0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.30889124387262124, -0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.36798765133638245, -0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.42561142339127539, -0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.48156756328753614, -0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0.53562066322056501, -0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, cmp);
}
#endif /* MODE_EX_HY */

#if defined (MODE_EX_HZ)
template<LayoutType layout_type>
void test1D_ExHz ()
{
  CoordinateType ct1 = CoordinateType::Y;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ex_cmp (overallSize, 1);
  cmp = &Ex_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (-0.21811863445648658, 0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.27908260941002716, 0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.3386998783352157, 0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.39727542433754848, 0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.45394742244500197, 0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.50918770413007142, 0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (-0.18730018750160604, 0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.24860062193325649, 0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.30889124387262124, 0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.36798765133638245, 0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.42561142339127539, 0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.48156756328753614, 0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (-0.53562066322056501, 0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ex_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, cmp);
}
#endif /* MODE_EX_HZ */

#if defined (MODE_EY_HX)
template<LayoutType layout_type>
void test1D_EyHx ()
{
  CoordinateType ct1 = CoordinateType::Z;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 0;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ey_cmp (overallSize, 1);
  cmp = &Ey_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0.21811863445648658, -0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.27908260941002716, -0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.3386998783352157, -0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.39727542433754848, -0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.45394742244500197, -0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.50918770413007142, -0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0.18730018750160604, -0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.24860062193325649, -0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.30889124387262124, -0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.36798765133638245, -0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.42561142339127539, -0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.48156756328753614, -0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.53562066322056501, -0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1, cmp);
}
#endif /* MODE_EY_HX */

#if defined (MODE_EY_HZ)
template<LayoutType layout_type>
void test1D_EyHz ()
{
  CoordinateType ct1 = CoordinateType::X;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ey_cmp (overallSize, 1);
  cmp = &Ey_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0.21811863445648658, -0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.27908260941002716, -0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.3386998783352157, -0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.39727542433754848, -0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.45394742244500197, -0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.50918770413007142, -0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0.18730018750160604, -0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.24860062193325649, -0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.30889124387262124, -0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.36798765133638245, -0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.42561142339127539, -0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.48156756328753614, -0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0.53562066322056501, -0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ey_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1, cmp);
}
#endif /* MODE_EY_HZ */

#if defined (MODE_EZ_HX)
template<LayoutType layout_type>
void test1D_EzHx ()
{
  CoordinateType ct1 = CoordinateType::Y;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ez_cmp (overallSize, 1);
  cmp = &Ez_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (-0.21811863445648658, 0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.27908260941002716, 0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.3386998783352157, 0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.39727542433754848, 0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.45394742244500197, 0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.50918770413007142, 0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (-0.18730018750160604, 0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.24860062193325649, 0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.30889124387262124, 0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.36798765133638245, 0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.42561142339127539, 0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.48156756328753614, 0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.53562066322056501, 0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, cmp);
}
#endif /* MODE_EZ_HX */

#if defined (MODE_EZ_HY)
template<LayoutType layout_type>
void test1D_EzHy ()
{
  CoordinateType ct1 = CoordinateType::X;

  GridCoordinate1D overallSize = GRID_COORDINATE_1D (SIZE, ct1);
  GridCoordinate1D pmlSize = GRID_COORDINATE_1D (PML_SIZE, ct1);
  GridCoordinate1D tfsfSizeLeft = GRID_COORDINATE_1D (TFSF_SIZE, ct1);
  GridCoordinate1D tfsfSizeRight = GRID_COORDINATE_1D (TFSF_SIZE, ct1);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> yeeLayout
    (overallSize,
     pmlSize,
     tfsfSizeLeft,
     tfsfSizeRight,
     angle1 * PhysicsConst::Pi / 180.0,
     angle2 * PhysicsConst::Pi / 180.0,
     angle3 * PhysicsConst::Pi / 180.0,
     useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, CoordinateType::NONE, CoordinateType::NONE);

  Grid<GridCoordinate1D> * cmp = NULLPTR;
#ifdef DOUBLE_VALUES
  Grid<GridCoordinate1D> Ez_cmp (overallSize, 1);
  cmp = &Ez_cmp;

  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == E_CENTERED)
  {
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (6, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (-0.21811863445648658, 0.97471655250974043), GRID_COORDINATE_1D (7, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.27908260941002716, 0.96196058514825045), GRID_COORDINATE_1D (8, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.3386998783352157, 0.93865257984534911), GRID_COORDINATE_1D (9, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.39727542433754848, 0.92054926754400479), GRID_COORDINATE_1D (10, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.45394742244500197, 0.88752443613856613), GRID_COORDINATE_1D (11, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.50918770413007142, 0.8648309348268548), GRID_COORDINATE_1D (12, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
  if (SOLVER_SETTINGS.getDoUseTFSF () && layout_type == H_CENTERED)
  {
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (0, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (1, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (2, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (3, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (4, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (5, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (-0.18730018750160604, 0.98158407156284588), GRID_COORDINATE_1D (6, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.24860062193325649, 0.96833856882899538), GRID_COORDINATE_1D (7, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.30889124387262124, 0.95030658249679933), GRID_COORDINATE_1D (8, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.36798765133638245, 0.92960092369467662), GRID_COORDINATE_1D (9, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.42561142339127539, 0.90403685184128513), GRID_COORDINATE_1D (10, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.48156756328753614, 0.8761776854827098), GRID_COORDINATE_1D (11, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (-0.53562066322056501, 0.84355796057397081), GRID_COORDINATE_1D (12, ct1), 0);

    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (13, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (14, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (15, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (16, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (17, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (18, ct1), 0);
    Ez_cmp.setFieldValue (FIELDVALUE (0, 0), GRID_COORDINATE_1D (19, ct1), 0);
  }
#endif /* DOUBLE_VALUES */

  test1D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, cmp);
}
#endif /* MODE_EZ_HY */

#if defined (MODE_TEX)
template<LayoutType layout_type>
void test2D_TEx ()
{
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}
#endif /* MODE_TEX */

#if defined (MODE_TEY)
template<LayoutType layout_type>
void test2D_TEy ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}
#endif /* MODE_TEY */

#if defined (MODE_TEZ)
template<LayoutType layout_type>
void test2D_TEz ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
}
#endif /* MODE_TEZ */

#if defined (MODE_TMX)
template<LayoutType layout_type>
void test2D_TMx ()
{
  CoordinateType ct1 = CoordinateType::Y;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (!intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEx (), yeeLayout.getMinExCoordFP (), ct1, ct2);
}
#endif /* MODE_TMX */

#if defined (MODE_TMY)
template<LayoutType layout_type>
void test2D_TMy ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Z;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 0;
  FPValue angle3 = 0;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (!intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (!intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEy (), yeeLayout.getMinEyCoordFP (), ct1, ct2);
}
#endif /* MODE_TMY */

#if defined (MODE_TMZ)
template<LayoutType layout_type>
void test2D_TMz ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;

  GridCoordinate2D overallSize = GRID_COORDINATE_2D (SIZE, SIZE, ct1, ct2);
  GridCoordinate2D pmlSize = GRID_COORDINATE_2D (PML_SIZE, PML_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeLeft = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);
  GridCoordinate2D tfsfSizeRight = GRID_COORDINATE_2D (TFSF_SIZE, TFSF_SIZE, ct1, ct2);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (!intScheme.getDoNeedEx ());
  ASSERT (!intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (!intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, CoordinateType::NONE);

  test2D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2);
}
#endif /* MODE_TMZ */

#if defined (MODE_DIM3)
template<LayoutType layout_type>
void test3D ()
{
  CoordinateType ct1 = CoordinateType::X;
  CoordinateType ct2 = CoordinateType::Y;
  CoordinateType ct3 = CoordinateType::Z;

  GridCoordinate3D overallSize = GRID_COORDINATE_3D (SIZE, SIZE, SIZE, ct1, ct2, ct3);
  GridCoordinate3D pmlSize = GRID_COORDINATE_3D (PML_SIZE, PML_SIZE, PML_SIZE, ct1, ct2, ct3);
  GridCoordinate3D tfsfSizeLeft = GRID_COORDINATE_3D (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);
  GridCoordinate3D tfsfSizeRight = GRID_COORDINATE_3D (TFSF_SIZE, TFSF_SIZE, TFSF_SIZE, ct1, ct2, ct3);

  bool useDoubleMaterialPrecision = false;

  FPValue angle1 = 90;
  FPValue angle2 = 90;
  FPValue angle3 = 90;

  YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> yeeLayout (overallSize,
                                                                        pmlSize,
                                                                        tfsfSizeLeft,
                                                                        tfsfSizeRight,
                                                                        angle1 * PhysicsConst::Pi / 180.0,
                                                                        angle2 * PhysicsConst::Pi / 180.0,
                                                                        angle3 * PhysicsConst::Pi / 180.0,
                                                                        useDoubleMaterialPrecision);

  InternalScheme<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> intScheme;
  intScheme.init (&yeeLayout, false);
  intScheme.initScheme (DX, LAMBDA);

  ASSERT (intScheme.getDoNeedEx ());
  ASSERT (intScheme.getDoNeedEy ());
  ASSERT (intScheme.getDoNeedEz ());
  ASSERT (intScheme.getDoNeedHx ());
  ASSERT (intScheme.getDoNeedHy ());
  ASSERT (intScheme.getDoNeedHz ());

  test<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type>
    (&intScheme, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, ct1, ct2, ct3);

  test3D (intScheme.getEz (), yeeLayout.getMinEzCoordFP (), ct1, ct2, ct3);
}
#endif /* MODE_DIM3 */

int main (int argc, char** argv)
{
  solverSettings.SetupFromCmd (argc, argv);

  /*
   * PML mode is not supported (Sigmas, Ca, Cb are not initialized here)
   */
  ASSERT (!solverSettings.getDoUsePML ());

#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ())
  {
    cudaCheckErrorCmd (cudaSetDevice(STOI (SOLVER_SETTINGS.getCudaGPUs ().c_str ())));
  }
#endif /* CUDA_ENABLED */

  solverSettings.Initialize ();

#if defined (MODE_EX_HY)
  test1D_ExHy<E_CENTERED> ();
  test1D_ExHy<H_CENTERED> ();
#endif /* MODE_EX_HY */
#if defined (MODE_EX_HZ)
  test1D_ExHz<E_CENTERED> ();
  test1D_ExHz<H_CENTERED> ();
#endif /* MODE_EX_HZ */
#if defined (MODE_EY_HX)
  test1D_EyHx<E_CENTERED> ();
  test1D_EyHx<H_CENTERED> ();
#endif /* MODE_EY_HX */
#if defined (MODE_EY_HZ)
  test1D_EyHz<E_CENTERED> ();
  test1D_EyHz<H_CENTERED> ();
#endif /* MODE_EY_HZ */
#if defined (MODE_EZ_HX)
  test1D_EzHx<E_CENTERED> ();
  test1D_EzHx<H_CENTERED> ();
#endif /* MODE_EZ_HX */
#if defined (MODE_EZ_HY)
  test1D_EzHy<E_CENTERED> ();
  test1D_EzHy<H_CENTERED> ();
#endif /* MODE_EZ_HY */

#if defined (MODE_TEX)
  test2D_TEx<E_CENTERED> ();
  test2D_TEx<H_CENTERED> ();
#endif /* MODE_TEX */
#if defined (MODE_TEY)
  test2D_TEy<E_CENTERED> ();
  test2D_TEy<H_CENTERED> ();
#endif /* MODE_TEY */
#if defined (MODE_TEZ)
  test2D_TEz<E_CENTERED> ();
  test2D_TEz<H_CENTERED> ();
#endif /* MODE_TEZ */
#if defined (MODE_TMX)
  test2D_TMx<E_CENTERED> ();
  test2D_TMx<H_CENTERED> ();
#endif /* MODE_TMX */
#if defined (MODE_TMY)
  test2D_TMy<E_CENTERED> ();
  test2D_TMy<H_CENTERED> ();
#endif /* MODE_TMY */
#if defined (MODE_TMZ)
  test2D_TMz<E_CENTERED> ();
  test2D_TMz<H_CENTERED> ();
#endif /* MODE_TMZ */

#if defined (MODE_DIM3)
  test3D<E_CENTERED> ();
  test3D<H_CENTERED> ();
#endif /* MODE_DIM3 */

  solverSettings.Uninitialize ();

  return 0;
} /* main */

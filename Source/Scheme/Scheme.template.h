/**
 * Perform computation of N steps for selected block
 *
 * NOTE: For GPU gpuIntSchemeOnGPU is used for all computations, i.e. where data in GPU memory is required.
 *       But for step to next time step gpuIntScheme is used for optimization (in order to not perform copy to/from GPU).
 *       So, grids in gpuIntScheme and gpuIntSchemeOnGPU are in a bit diverged states. Make sure to correctly use them!
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performNStepsForBlock (time_step tStart, /**< start time step */
                                                          time_step N, /**< number of time steps to compute */
                                                          TC blockIdx) /**< index of block, for which computations are to be performed */
{
#ifdef CUDA_ENABLED
  /*
   * Copy InternalScheme to GPU
   */
  gpuIntScheme->copyFromCPU (blockIdx * blockSize, blockSize);
  gpuIntSchemeOnGPU->copyToGPU (gpuIntScheme);
  cudaCheckErrorCmd (cudaMemcpy (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));
#endif /* CUDA_ENABLED */

  for (time_step t = tStart; t < tStart + N; ++t)
  {
    DPRINTF (LOG_LEVEL_NONE, "calculating time step %d\n", t);

#ifdef CUDA_ENABLED
    TC ExStart = gpuIntScheme->doNeedEx ? gpuIntScheme->Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = gpuIntScheme->doNeedEx ? gpuIntScheme->Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = gpuIntScheme->doNeedEy ? gpuIntScheme->Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = gpuIntScheme->doNeedEy ? gpuIntScheme->Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = gpuIntScheme->doNeedEz ? gpuIntScheme->Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = gpuIntScheme->doNeedEz ? gpuIntScheme->Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = gpuIntScheme->doNeedHx ? gpuIntScheme->Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = gpuIntScheme->doNeedHx ? gpuIntScheme->Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = gpuIntScheme->doNeedHy ? gpuIntScheme->Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = gpuIntScheme->doNeedHy ? gpuIntScheme->Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = gpuIntScheme->doNeedHz ? gpuIntScheme->Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = gpuIntScheme->doNeedHz ? gpuIntScheme->Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
#else /* CUDA_ENABLED */
    TC ExStart = intScheme->doNeedEx ? intScheme->Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = intScheme->doNeedEx ? intScheme->Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = intScheme->doNeedEy ? intScheme->Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = intScheme->doNeedEy ? intScheme->Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = intScheme->doNeedEz ? intScheme->Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = intScheme->doNeedEz ? intScheme->Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = intScheme->doNeedHx ? intScheme->Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = intScheme->doNeedHx ? intScheme->Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = intScheme->doNeedHy ? intScheme->Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = intScheme->doNeedHy ? intScheme->Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = intScheme->doNeedHz ? intScheme->Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = intScheme->doNeedHz ? intScheme->Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
#endif /* CUDA_ENABLED */

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->performPlaneWaveEStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getEInc ()->getSize ());
      gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchEInc (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEInc ()->nextTimeStep ();
#else /* CUDA_ENABLED */
      intScheme->performPlaneWaveESteps (t, zero1D, intScheme->getEInc ()->getSize ());
      intScheme->getEInc ()->shiftInTime ();
      intScheme->getEInc ()->nextTimeStep ();
#endif /* !CUDA_ENABLED */
    }

    if (intScheme->getDoNeedEx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEx (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1x (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1x ()->nextTimeStep ();
      }
#else
      intScheme->getEx ()->shiftInTime ();
      intScheme->getEx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDx ()->shiftInTime ();
        intScheme->getDx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1x ()->shiftInTime ();
        intScheme->getD1x ()->nextTimeStep ();
      }
#endif
    }

    if (intScheme->getDoNeedEy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEy (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1y (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1y ()->nextTimeStep ();
      }
#else
      intScheme->getEy ()->shiftInTime ();
      intScheme->getEy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDy ()->shiftInTime ();
        intScheme->getDy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1y ()->shiftInTime ();
        intScheme->getD1y ()->nextTimeStep ();
      }
#endif
    }

    if (intScheme->getDoNeedEz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEz (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getEz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getDz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1z (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getD1z ()->nextTimeStep ();
      }
#else
      intScheme->getEz ()->shiftInTime ();
      intScheme->getEz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getDz ()->shiftInTime ();
        intScheme->getDz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getD1z ()->shiftInTime ();
        intScheme->getD1z ()->nextTimeStep ();
      }
#endif
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->performPlaneWaveHStepsKernelLaunch (d_gpuIntSchemeOnGPU, t, zero1D, gpuIntScheme->getHInc ()->getSize ());
      gpuIntSchemeOnGPU->shiftInTimePlaneWaveKernelLaunchHInc (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHInc ()->nextTimeStep ();
#else /* CUDA_ENABLED */
      intScheme->performPlaneWaveHSteps (t, zero1D, intScheme->getHInc ()->getSize ());
      intScheme->getHInc ()->shiftInTime ();
      intScheme->getHInc ()->nextTimeStep ();
#endif /* !CUDA_ENABLED */
    }

    if (intScheme->getDoNeedHx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHx (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1x (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1x ()->nextTimeStep ();
      }
#else
      intScheme->getHx ()->shiftInTime ();
      intScheme->getHx ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBx ()->shiftInTime ();
        intScheme->getBx ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1x ()->shiftInTime ();
        intScheme->getB1x ()->nextTimeStep ();
      }
#endif
    }

    if (intScheme->getDoNeedHy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHy (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1y (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1y ()->nextTimeStep ();
      }
#else
      intScheme->getHy ()->shiftInTime ();
      intScheme->getHy ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBy ()->shiftInTime ();
        intScheme->getBy ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1y ()->shiftInTime ();
        intScheme->getB1y ()->nextTimeStep ();
      }
#endif
    }

    if (intScheme->getDoNeedHz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);

#ifdef CUDA_ENABLED
      gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHz (d_gpuIntSchemeOnGPU);
      gpuIntScheme->getHz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getBz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1z (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getB1z ()->nextTimeStep ();
      }
#else
      intScheme->getHz ()->shiftInTime ();
      intScheme->getHz ()->nextTimeStep ();

      if (SOLVER_SETTINGS.getDoUsePML ())
      {
        intScheme->getBz ()->shiftInTime ();
        intScheme->getBz ()->nextTimeStep ();
      }
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        intScheme->getB1z ()->shiftInTime ();
        intScheme->getB1z ()->nextTimeStep ();
      }
#endif
    }
  }

#ifdef CUDA_ENABLED
  /*
   * Copy back from GPU to CPU
   */
  bool finalCopy = blockIdx + TC_COORD (1, 1, 1, ct1, ct2, ct3) == blockCount;
  gpuIntScheme->copyBackToCPU (NTimeSteps, finalCopy);
#endif /* CUDA_ENABLED */
}

/**
 * Perform share operations, required for grids
 *
 * NOTE: this basically should be non-empty for ParallelGrids only
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::share ()
{
#ifdef PARALLEL_GRID
  if (!useParallel)
  {
    return;
  }

  if (intScheme->getDoNeedEx ())
  {
    ASSERT (((ParallelGrid *) intScheme->Ex)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Ex)->share ();
    ((ParallelGrid *) intScheme->Ex)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->Dx)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->Dx)->share ();
      ((ParallelGrid *) intScheme->Dx)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->D1x)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->D1x)->share ();
      ((ParallelGrid *) intScheme->D1x)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedEy ())
  {
    ASSERT (((ParallelGrid *) intScheme->Ey)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Ey)->share ();
    ((ParallelGrid *) intScheme->Ey)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->Dy)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->Dy)->share ();
      ((ParallelGrid *) intScheme->Dy)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->D1y)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->D1y)->share ();
      ((ParallelGrid *) intScheme->D1y)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedEz ())
  {
    ASSERT (((ParallelGrid *) intScheme->Ez)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Ez)->share ();
    ((ParallelGrid *) intScheme->Ez)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->Dz)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->Dz)->share ();
      ((ParallelGrid *) intScheme->Dz)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->D1z)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->D1z)->share ();
      ((ParallelGrid *) intScheme->D1z)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHx ())
  {
    ASSERT (((ParallelGrid *) intScheme->Hx)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Hx)->share ();
    ((ParallelGrid *) intScheme->Hx)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->Bx)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->Bx)->share ();
      ((ParallelGrid *) intScheme->Bx)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->B1x)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->B1x)->share ();
      ((ParallelGrid *) intScheme->B1x)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHy ())
  {
    ASSERT (((ParallelGrid *) intScheme->Hy)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Hy)->share ();
    ((ParallelGrid *) intScheme->Hy)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->By)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->By)->share ();
      ((ParallelGrid *) intScheme->By)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->B1y)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->B1y)->share ();
      ((ParallelGrid *) intScheme->B1y)->zeroShareStep ();
    }
  }

  if (intScheme->getDoNeedHz ())
  {
    ASSERT (((ParallelGrid *) intScheme->Hz)->getShareStep () == NTimeSteps);
    ((ParallelGrid *) intScheme->Hz)->share ();
    ((ParallelGrid *) intScheme->Hz)->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (((ParallelGrid *) intScheme->Bz)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->Bz)->share ();
      ((ParallelGrid *) intScheme->Bz)->zeroShareStep ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ASSERT (((ParallelGrid *) intScheme->B1z)->getShareStep () == NTimeSteps);
      ((ParallelGrid *) intScheme->B1z)->share ();
      ((ParallelGrid *) intScheme->B1z)->zeroShareStep ();
    }
  }
#endif /* PARALLEL_GRID */
}

/**
 * Perform balancing operations
 *
 * NOTE: this should be non-empty basically for ParallelGrids only
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::rebalance ()
{

}

/**
 * Perform computations of single time step for specific field and for specified chunk.
 *
 * NOTE: Start and End coordinates should correctly consider buffers in parallel grid,
 *       which means, that computations should not be performed for incorrect grid points.
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
void
Scheme<Type, TCoord, layout_type>::performFieldSteps (time_step t, /**< time step to compute */
                                                      TC Start, /**< start coordinate of chunk to compute */
                                                      TC End) /**< end coordinate of chunk to compute */
{
  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      calculateFieldStep<grid_type, true, true> (t, Start, End);
    }
    else
    {
      calculateFieldStep<grid_type, true, false> (t, Start, End);
    }
  }
  else
  {
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      calculateFieldStep<grid_type, false, true> (t, Start, End);
    }
    else
    {
      calculateFieldStep<grid_type, false, false> (t, Start, End);
    }
  }

  bool doUsePointSource;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEx ();
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEy ();
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEz ();
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHx ();
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHy ();
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHz ();
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (doUsePointSource)
  {
    performPointSourceCalc<grid_type> (t);
  }
}

/**
 * Perform computations of single time step for specific field and for specified chunk for PML/metamaterials modes.
 *
 * NOTE: Start and End coordinates should correctly consider buffers in parallel grid,
 *       which means, that computations should not be performed for incorrect grid points.
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStep (time_step t, /**< time step to calculate */
                                                       TC start, /**< start coordinate of chunk to perform computations on */
                                                       TC end) /**< end coordinate of chunk to perform computations on */
{
  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

  FPValue k_mod = FPValue (1);

  Grid<TC> *grid = NULLPTR;
  GridType gridType = GridType::NONE;

  Grid<TC> *materialGrid = NULLPTR;
  GridType materialGridType = GridType::NONE;

  Grid<TC> *materialGrid1 = NULLPTR;
  GridType materialGridType1 = GridType::NONE;

  Grid<TC> *materialGrid2 = NULLPTR;
  GridType materialGridType2 = GridType::NONE;

  Grid<TC> *materialGrid3 = NULLPTR;
  GridType materialGridType3 = GridType::NONE;

  Grid<TC> *materialGrid4 = NULLPTR;
  GridType materialGridType4 = GridType::NONE;

  Grid<TC> *materialGrid5 = NULLPTR;
  GridType materialGridType5 = GridType::NONE;

  Grid<TC> *oppositeGrid1 = NULLPTR;
  Grid<TC> *oppositeGrid2 = NULLPTR;

  Grid<TC> *gridPML1 = NULLPTR;
  GridType gridPMLType1 = GridType::NONE;

  Grid<TC> *gridPML2 = NULLPTR;
  GridType gridPMLType2 = GridType::NONE;

  Grid<TC> *Ca = NULLPTR;
  Grid<TC> *Cb = NULLPTR;

  Grid<TC> *CB0 = NULLPTR;
  Grid<TC> *CB1 = NULLPTR;
  Grid<TC> *CB2 = NULLPTR;
  Grid<TC> *CA1 = NULLPTR;
  Grid<TC> *CA2 = NULLPTR;

  Grid<TC> *CaPML = NULLPTR;
  Grid<TC> *CbPML = NULLPTR;
  Grid<TC> *CcPML = NULLPTR;

  SourceCallBack rightSideFunc = NULLPTR;
  SourceCallBack borderFunc = NULLPTR;
  SourceCallBack exactFunc = NULLPTR;

  TCS diff11;
  TCS diff12;
  TCS diff21;
  TCS diff22;

  /*
   * TODO: remove this, multiply on this at initialization
   */
  FPValue materialModifier;

  intScheme->calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&grid, &gridType,
    &materialGrid, &materialGridType, &materialGrid1, &materialGridType1, &materialGrid2, &materialGridType2,
    &materialGrid3, &materialGridType3, &materialGrid4, &materialGridType4, &materialGrid5, &materialGridType5,
    &oppositeGrid1, &oppositeGrid2, &gridPML1, &gridPMLType1, &gridPML2, &gridPMLType2,
    &rightSideFunc, &borderFunc, &exactFunc, &materialModifier, &Ca, &Cb,
    &CB0, &CB1, &CB2, &CA1, &CA2, &CaPML, &CbPML, &CcPML);

  intScheme->calculateFieldStepInitDiff<grid_type> (&diff11, &diff12, &diff21, &diff22);

#ifdef CUDA_ENABLED
  CudaGrid<TC> *d_grid = NULLPTR;
  GridType _gridType = GridType::NONE;

  CudaGrid<TC> *d_materialGrid = NULLPTR;
  GridType _materialGridType = GridType::NONE;

  CudaGrid<TC> *d_materialGrid1 = NULLPTR;
  GridType _materialGridType1 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid2 = NULLPTR;
  GridType _materialGridType2 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid3 = NULLPTR;
  GridType _materialGridType3 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid4 = NULLPTR;
  GridType _materialGridType4 = GridType::NONE;

  CudaGrid<TC> *d_materialGrid5 = NULLPTR;
  GridType _materialGridType5 = GridType::NONE;

  CudaGrid<TC> *d_oppositeGrid1 = NULLPTR;
  CudaGrid<TC> *d_oppositeGrid2 = NULLPTR;

  CudaGrid<TC> *d_gridPML1 = NULLPTR;
  GridType _gridPMLType1 = GridType::NONE;

  CudaGrid<TC> *d_gridPML2 = NULLPTR;
  GridType _gridPMLType2 = GridType::NONE;

  CudaGrid<TC> *d_Ca = NULLPTR;
  CudaGrid<TC> *d_Cb = NULLPTR;

  Grid<TC> *d_CB0 = NULLPTR;
  Grid<TC> *d_CB1 = NULLPTR;
  Grid<TC> *d_CB2 = NULLPTR;
  Grid<TC> *d_CA1 = NULLPTR;
  Grid<TC> *d_CA2 = NULLPTR;

  Grid<TC> *d_CaPML = NULLPTR;
  Grid<TC> *d_CbPML = NULLPTR;
  Grid<TC> *d_CcPML = NULLPTR;

  SourceCallBack _rightSideFunc = NULLPTR;
  SourceCallBack _borderFunc = NULLPTR;
  SourceCallBack _exactFunc = NULLPTR;

  FPValue _materialModifier;

  gpuIntSchemeOnGPU->template calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&d_grid, &_gridType,
    &d_materialGrid, &_materialGridType, &d_materialGrid1, &_materialGridType1, &d_materialGrid2, &_materialGridType2,
    &d_materialGrid3, &_materialGridType3, &d_materialGrid4, &_materialGridType4, &d_materialGrid5, &_materialGridType5,
    &d_oppositeGrid1, &d_oppositeGrid2, &d_gridPML1, &_gridPMLType1, &d_gridPML2, &_gridPMLType2,
    &_rightSideFunc, &_borderFunc, &_exactFunc, &_materialModifier, &d_Ca, &d_Cb,
    &d_CB0, &d_CB1, &d_CB2, &d_CA1, &d_CA2, &d_CaPML, &d_CbPML, &d_CcPML);

  // TODO: support right side func for CUDA
  _rightSideFunc = NULLPTR;

#ifdef ENABLE_ASSERTS
  TCS _diff11;
  TCS _diff12;
  TCS _diff21;
  TCS _diff22;

  gpuIntSchemeOnGPU->calculateFieldStepInitDiff<grid_type> (&diff11, &diff12, &diff21, &diff22);
  ASSERT (diff11 == _diff11 && diff12 == _diff12 && diff21 == _diff21 && diff22 == _diff22);
#endif /* ENABLE_ASSERTS */

#endif /* CUDA_ENABLED */

  // TODO: specialize for each dimension
  GridCoordinate3D start3D;
  GridCoordinate3D end3D;

  expandTo3DStartEnd (start, end, start3D, end3D, ct1, ct2, ct3);

  // TODO: remove this check for each iteration
  if (t > 0)
  {
#ifdef CUDA_ENABLED

    // Launch kernel here
    gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <grid_type> (d_gpuIntSchemeOnGPU, start3D, end3D,
                                                                            t, diff11, diff12, diff21, diff22,
                                                                            d_grid,
                                                                            d_oppositeGrid1, d_oppositeGrid2, _rightSideFunc, d_Ca, d_Cb,
                                                                            usePML,
                                                                            gridType, materialGrid, materialGridType,
                                                                            materialModifier,
                                                                            SOLVER_SETTINGS.getDoUseCaCbGrids ());

#else /* CUDA_ENABLED */

    for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
      {
        // TODO: check that this is optimized out in case 2D mode
        for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);

          // TODO: add getTotalPositionDiff here, which will be called before loop
          TC posAbs = grid->getTotalPosition (pos);

          TCFP coordFP;

          if (rightSideFunc != NULLPTR)
          {
            switch (grid_type)
            {
              case (static_cast<uint8_t> (GridType::EX)):
              {
                coordFP = yeeLayout->getExCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::EY)):
              {
                coordFP = yeeLayout->getEyCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::EZ)):
              {
                coordFP = yeeLayout->getEzCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HX)):
              {
                coordFP = yeeLayout->getHxCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HY)):
              {
                coordFP = yeeLayout->getHyCoordFP (posAbs);
                break;
              }
              case (static_cast<uint8_t> (GridType::HZ)):
              {
                coordFP = yeeLayout->getHzCoordFP (posAbs);
                break;
              }
              default:
              {
                UNREACHABLE;
              }
            }
          }

          if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
          {
            intScheme->calculateFieldStepIteration<grid_type, true> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               grid, coordFP,
                                                               oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                               usePML,
                                                               gridType, materialGrid, materialGridType,
                                                               materialModifier);
          }
          else
          {
            intScheme->calculateFieldStepIteration<grid_type, false> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               grid, coordFP,
                                                               oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                               usePML,
                                                               gridType, materialGrid, materialGridType,
                                                               materialModifier);
          }
        }
      }
    }
#endif

    if (usePML)
    {
      if (useMetamaterials)
      {
#ifdef CUDA_ENABLED

        // Launch kernel here
        gpuIntSchemeOnGPU->calculateFieldStepIterationPMLMetamaterialsKernelLaunch
          (d_gpuIntSchemeOnGPU, start3D, end3D,
           t, pos, grid, gridPML1,
           CB0, CB1, CB2, CA1, CA2,
           gridType,
           materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
           materialModifier);

#else /* CUDA_ENABLED */

        for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
          {
            // TODO: check that this loop is optimized out
            for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
            {
              TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
              intScheme->calculateFieldStepIterationPMLMetamaterials (t, pos, grid, gridPML1,
                CB0, CB1, CB2, CA1, CA2,
                gridType,
                materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
                materialModifier);
            }
          }
        }

#endif /* !CUDA_ENABLED */
      }

#ifdef CUDA_ENABLED

      gpuIntSchemeOnGPU->template calculateFieldStepIterationPMLKernelLaunch <useMetamaterials> (d_gpuIntSchemeOnGPU, start3D, end3D,
        t, pos, grid, gridPML1, gridPML2, CaPML, CbPML, CcPML, gridPMLType1,
        materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
        materialModifier);

#else /* CUDA_ENABLED */

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            intScheme->template calculateFieldStepIterationPML<useMetamaterials> (t, pos, grid, gridPML1, gridPML2, CaPML, CbPML, CcPML, gridPMLType1,
              materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
              materialModifier);
          }
        }
      }

#endif /* !CUDA_ENABLED */
    }
  }

#ifndef CUDA_ENABLED
  // TODO: support border func, exact func and right side func for CUDA

  if (borderFunc != NULLPTR)
  {
    GridCoordinate3D startBorder;
    GridCoordinate3D endBorder;

    expandTo3DStartEnd (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3),
                        grid->getSize (),
                        startBorder,
                        endBorder,
                        ct1, ct2, ct3);

    for (grid_coord i = startBorder.get1 (); i < endBorder.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startBorder.get2 (); j < endBorder.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startBorder.get3 (); k < endBorder.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          intScheme->calculateFieldStepIterationBorder<grid_type> (t, pos, grid, borderFunc);
        }
      }
    }
  }

  if (exactFunc != NULLPTR)
  {
    FPValue normRe = 0.0;
    FPValue normIm = 0.0;
    FPValue normMod = 0.0;

    FPValue maxRe = 0.0;
    FPValue maxIm = 0.0;
    FPValue maxMod = 0.0;

    GridCoordinate3D startNorm = start3D;
    GridCoordinate3D endNorm = end3D;

    if (SOLVER_SETTINGS.getExactSolutionCompareStartX () != 0)
    {
      startNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareStartX ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareStartY () != 0)
    {
      startNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareStartY ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareStartZ () != 0)
    {
      startNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareStartZ ());
    }

    if (SOLVER_SETTINGS.getExactSolutionCompareEndX () != 0)
    {
      endNorm.set1 (SOLVER_SETTINGS.getExactSolutionCompareEndX ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareEndY () != 0)
    {
      endNorm.set2 (SOLVER_SETTINGS.getExactSolutionCompareEndY ());
    }
    if (SOLVER_SETTINGS.getExactSolutionCompareEndZ () != 0)
    {
      endNorm.set3 (SOLVER_SETTINGS.getExactSolutionCompareEndZ ());
    }

    IGRID<TC> *normGrid = grid;
    if (usePML)
    {
      grid = gridPML2;
    }

    for (grid_coord i = startNorm.get1 (); i < endNorm.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startNorm.get2 (); j < endNorm.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startNorm.get3 (); k < endNorm.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          intScheme->calculateFieldStepIterationExact<grid_type> (t, pos, grid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
        }
      }
    }

#ifdef COMPLEX_FIELD_VALUES
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
    normIm = sqrt (normIm / grid->getSize ().calculateTotalCoord ());
    normMod = sqrt (normMod / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
      grid->getName (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
      grid->getName (), t, normRe, normRe * 100.0 / maxRe);
#endif
  }

#endif /* !CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initBlocks (time_step t_total)
{
  totalTimeSteps = t_total;

  /*
   * TODO: currently only single block is set up here, but underlying methods should support more?
   */
  blockCount = TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

  // TODO: allocate previous step storage for cuda blocks (see page 81)

#ifdef PARALLEL_GRID
  ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;
  blockSize = parallelYeeLayout->getSizeForCurNode ();
#else
  blockSize = yeeLayout->getSize ();
#endif

#ifdef PARALLEL_GRID
  if (useParallel)
  {
    time_step parallelBuf = (time_step) SOLVER_SETTINGS.getBufferSize ();
    NTimeSteps = parallelBuf;
  }
  else
#endif /* PARALLEL_GRID */
  {
    NTimeSteps = totalTimeSteps;
  }

#ifdef CUDA_ENABLED
  if (blockCount.calculateTotalCoord () > 1)
  {
    /*
     * More than one block is used, have to consider buffers now
     */
    time_step cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();

#ifdef PARALLEL_GRID
    if (useParallel)
    {
      /*
       * Cuda grid buffer can't be greater than parallel grid buffer, because there will be no data to fill it with.
       * If cuda grid buffer is less than parallel grid buffer, then parallel grid buffer won't be used fully, which
       * is undesirable. So, restrict buffers to be identical for the case of both parallel mode and cuda mode.
       */
      ALWAYS_ASSERT (cudaBuf == (time_step) SOLVER_SETTINGS.getBufferSize ())
    }
#endif /* PARALLEL_GRID */

    NTimeSteps = cudaBuf;
  }

  /*
   * Init InternalScheme on GPU
   */
  time_step cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();

  gpuIntScheme = new InternalSchemeGPU<Type, TCoord, layout_type> ();
  gpuIntSchemeOnGPU = new InternalSchemeGPU<Type, TCoord, layout_type> ();

  gpuIntScheme->initFromCPU (this, blockSize, TC_COORD (cudaBuf, cudaBuf, cudaBuf, ct1, ct2, ct3));
  gpuIntSchemeOnGPU->initOnGPU (gpuIntScheme);

  cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>)));
#endif /* CUDA_ENABLED */
}

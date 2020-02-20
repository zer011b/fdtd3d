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
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ()
      && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
  {
    /*
     * Copy InternalScheme to GPU
     */
    gpuIntScheme->copyFromCPU (blockIdx * blockSize, blockSize);
    gpuIntSchemeOnGPU->copyToGPU (gpuIntScheme);
    cudaCheckErrorCmd (cudaMemcpy (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));
  }
#endif /* CUDA_ENABLED */

  for (time_step t = tStart; t < tStart + N; ++t)
  {
    if (processId == 0)
    {
      DPRINTF (LOG_LEVEL_STAGES, "Calculating time step %u...\n", t);
    }

    TC ExStart, ExEnd;
    TC EyStart, EyEnd;
    TC EzStart, EzEnd;
    TC HxStart, HxEnd;
    TC HyStart, HyEnd;
    TC HzStart, HzEnd;

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      ExStart = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      ExEnd = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      EyStart = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      EyEnd = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      EzStart = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      EzEnd = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HxStart = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HxEnd = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HyStart = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HyEnd = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HzStart = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HzEnd = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    }
    else
#endif /* CUDA_ENABLED */
    {
      ExStart = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      ExEnd = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      EyStart = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      EyEnd = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      EzStart = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      EzEnd = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HxStart = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HxEnd = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HyStart = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HyEnd = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

      HzStart = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
      HzEnd = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
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

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of dynamic grid. "
                      "Recompile it with -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedEx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
    }

    if (intScheme->getDoNeedEy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
    }

    if (intScheme->getDoNeedEz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of dynamic grid. "
                      "Recompile it with -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedEx ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEx ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDx (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getDx ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1x (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getD1x ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getEx ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getDx ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getD1x ()->shiftInTime ();
        }
      }
    }

    if (intScheme->getDoNeedEy ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEy ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDy (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getDy ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1y (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getD1y ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getEy ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getDy ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getD1y ()->shiftInTime ();
        }
      }
    }

    if (intScheme->getDoNeedEz ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchEz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getEz ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchDz (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getDz ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchD1z (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getD1z ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getEz ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getDz ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getD1z ()->shiftInTime ();
        }
      }
    }

#ifdef PARALLEL_GRID
#ifdef CUDA_ENABLED
    if (!SOLVER_SETTINGS.getDoUseCuda ())
#endif
    {
      tryShareE ();
    }
#endif /* PARALLEL_GRID */

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D = GRID_COORDINATE_1D (0, CoordinateType::X);

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
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

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of dynamic grid. "
                      "Recompile it with -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedHx ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
    }

    if (intScheme->getDoNeedHy ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
    }

    if (intScheme->getDoNeedHz ())
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of dynamic grid. "
                      "Recompile it with -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedHx ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHx (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHx ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBx (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getBx ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1x (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getB1x ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getHx ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getBx ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getB1x ()->shiftInTime ();
        }
      }
    }

    if (intScheme->getDoNeedHy ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHy (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHy ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBy (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getBy ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1y (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getB1y ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getHy ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getBy ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getB1y ()->shiftInTime ();
        }
      }
    }

    if (intScheme->getDoNeedHz ())
    {
#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->shiftInTimeKernelLaunchHz (d_gpuIntSchemeOnGPU);
        gpuIntScheme->getHz ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchBz (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getBz ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          gpuIntSchemeOnGPU->shiftInTimeKernelLaunchB1z (d_gpuIntSchemeOnGPU);
          gpuIntScheme->getB1z ()->shiftInTime ();
        }
      }
      else
#endif
      {
        intScheme->getHz ()->shiftInTime ();

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          intScheme->getBz ()->shiftInTime ();
        }
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          intScheme->getB1z ()->shiftInTime ();
        }
      }
    }

#ifdef PARALLEL_GRID
#ifdef CUDA_ENABLED
    if (!SOLVER_SETTINGS.getDoUseCuda ())
#endif
    {
      tryShareH ();
    }
#endif /* PARALLEL_GRID */
  }

#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ()
      && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
  {
    /*
     * Copy back from GPU to CPU
     */
    // TODO: finalCopy is setup incorrectly for multidim case of block splitting
    bool finalCopy = blockIdx + TC_COORD (1, 1, 1, ct1, ct2, ct3) == blockCount;
    gpuIntScheme->copyBackToCPU (NTimeSteps, finalCopy);
  }
#endif /* CUDA_ENABLED */
}

#ifdef PARALLEL_GRID
/**
 * Perform share operations with checks
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::tryShareE ()
{
  if (!useParallel)
  {
    return;
  }

  eGroup->nextShareStep ();

  if (eGroup->isShareTime ())
  {
    ASSERT (eGroup->getShareStep () == NTimeSteps);

    shareE ();
  }
}

/**
 * Perform share operations with checks
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::tryShareH ()
{
  if (!useParallel)
  {
    return;
  }

  hGroup->nextShareStep ();

  if (hGroup->isShareTime ())
  {
    ASSERT (hGroup->getShareStep () == NTimeSteps);

    shareH ();
  }
}

/**
 * Perform share operations, required for grids
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::shareE ()
{
  if (!useParallel)
  {
    return;
  }

  if (intScheme->getDoNeedEx ())
  {
    ((ParallelGrid *) intScheme->getEx ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getDx ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getD1x ())->share ();
    }
  }

  if (intScheme->getDoNeedEy ())
  {
    ((ParallelGrid *) intScheme->getEy ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getDy ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getD1y ())->share ();
    }
  }

  if (intScheme->getDoNeedEz ())
  {
    ((ParallelGrid *) intScheme->getEz ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getDz ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getD1z ())->share ();
    }
  }

  eGroup->zeroShareStep ();
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::shareH ()
{
  if (!useParallel)
  {
    return;
  }

  if (intScheme->getDoNeedHx ())
  {
    ((ParallelGrid *) intScheme->getHx ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getBx ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getB1x ())->share ();
    }
  }

  if (intScheme->getDoNeedHy ())
  {
    ((ParallelGrid *) intScheme->getHy ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getBy ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getB1y ())->share ();
    }
  }

  if (intScheme->getDoNeedHz ())
  {
    ((ParallelGrid *) intScheme->getHz ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ((ParallelGrid *) intScheme->getBz ())->share ();
    }
    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) intScheme->getB1z ())->share ();
    }
  }

  hGroup->zeroShareStep ();
}
#endif /* PARALLEL_GRID */

/**
 * Perform balancing operations
 *
 * NOTE: this should be non-empty basically for ParallelGrids only
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::rebalance ()
{
  // if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
  // {
  // #if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
  //   //if (false && t % SOLVER_SETTINGS.getRebalanceStep () == 0)
  //   if (t % diffT == 0 && t > 0)
  //   {
  //     if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
  //     {
  //       DPRINTF (LOG_LEVEL_STAGES, "Try rebalance on step %u, steps elapsed after previous %u\n", t, diffT);
  //     }
  //
  //     ASSERT (isParallelLayout);
  //
  //     ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;
  //
  //     if (parallelYeeLayout->Rebalance (diffT))
  //     {
  //       DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalancing for process %d!\n", ParallelGrid::getParallelCore ()->getProcessId ());
  //
  //       ((ParallelGrid *) internalScheme.Eps)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //       ((ParallelGrid *) internalScheme.Mu)->Resize (parallelYeeLayout->getMuSizeForCurNode ());
  //
  //       if (internalScheme.doNeedEx)
  //       {
  //         ((ParallelGrid *) internalScheme.Ex)->Resize (parallelYeeLayout->getExSizeForCurNode ());
  //       }
  //       if (internalScheme.doNeedEy)
  //       {
  //         ((ParallelGrid *) internalScheme.Ey)->Resize (parallelYeeLayout->getEySizeForCurNode ());
  //       }
  //       if (internalScheme.doNeedEz)
  //       {
  //         ((ParallelGrid *) internalScheme.Ez)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
  //       }
  //
  //       if (internalScheme.doNeedHx)
  //       {
  //         ((ParallelGrid *) internalScheme.Hx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
  //       }
  //       if (internalScheme.doNeedHy)
  //       {
  //         ((ParallelGrid *) internalScheme.Hy)->Resize (parallelYeeLayout->getHySizeForCurNode ());
  //       }
  //       if (internalScheme.doNeedHz)
  //       {
  //         ((ParallelGrid *) internalScheme.Hz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
  //       }
  //
  //       if (SOLVER_SETTINGS.getDoUsePML ())
  //       {
  //         if (internalScheme.doNeedEx)
  //         {
  //           ((ParallelGrid *) internalScheme.Dx)->Resize (parallelYeeLayout->getExSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedEy)
  //         {
  //           ((ParallelGrid *) internalScheme.Dy)->Resize (parallelYeeLayout->getEySizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedEz)
  //         {
  //           ((ParallelGrid *) internalScheme.Dz)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
  //         }
  //
  //         if (internalScheme.doNeedHx)
  //         {
  //           ((ParallelGrid *) internalScheme.Bx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedHy)
  //         {
  //           ((ParallelGrid *) internalScheme.By)->Resize (parallelYeeLayout->getHySizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedHz)
  //         {
  //           ((ParallelGrid *) internalScheme.Bz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
  //         }
  //
  //         if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  //         {
  //           if (internalScheme.doNeedEx)
  //           {
  //             ((ParallelGrid *) internalScheme.D1x)->Resize (parallelYeeLayout->getExSizeForCurNode ());
  //           }
  //           if (internalScheme.doNeedEy)
  //           {
  //             ((ParallelGrid *) internalScheme.D1y)->Resize (parallelYeeLayout->getEySizeForCurNode ());
  //           }
  //           if (internalScheme.doNeedEz)
  //           {
  //             ((ParallelGrid *) internalScheme.D1z)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
  //           }
  //
  //           if (internalScheme.doNeedHx)
  //           {
  //             ((ParallelGrid *) internalScheme.B1x)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
  //           }
  //           if (internalScheme.doNeedHy)
  //           {
  //             ((ParallelGrid *) internalScheme.B1y)->Resize (parallelYeeLayout->getHySizeForCurNode ());
  //           }
  //           if (internalScheme.doNeedHz)
  //           {
  //             ((ParallelGrid *) internalScheme.B1z)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
  //           }
  //         }
  //
  //         if (internalScheme.doNeedSigmaX)
  //         {
  //           ((ParallelGrid *) internalScheme.SigmaX)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedSigmaY)
  //         {
  //           ((ParallelGrid *) internalScheme.SigmaY)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedSigmaZ)
  //         {
  //           ((ParallelGrid *) internalScheme.SigmaZ)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         }
  //       }
  //
  //       if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  //       {
  //         ((ParallelGrid *) internalScheme.OmegaPE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         ((ParallelGrid *) internalScheme.GammaE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         ((ParallelGrid *) internalScheme.OmegaPM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //         ((ParallelGrid *) internalScheme.GammaM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
  //       }
  //
  //       //diffT += 1;
  //       //diffT *= 2;
  //     }
  //   }
  // #else
  //   ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
  //                   "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
  // #endif
  // }
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
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      gpuIntSchemeOnGPU->template performPointSourceCalcKernelLaunch<grid_type> (d_gpuIntSchemeOnGPU, t);
    }
    else
#endif /* CUDA_ENABLED */
    {
      intScheme->template performPointSourceCalc<grid_type> (t);
    }
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
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

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

  intScheme->template calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&grid, &gridType,
    &materialGrid, &materialGridType, &materialGrid1, &materialGridType1, &materialGrid2, &materialGridType2,
    &materialGrid3, &materialGridType3, &materialGrid4, &materialGridType4, &materialGrid5, &materialGridType5,
    &oppositeGrid1, &oppositeGrid2, &gridPML1, &gridPMLType1, &gridPML2, &gridPMLType2,
    &rightSideFunc, &borderFunc, &exactFunc, &materialModifier, &Ca, &Cb,
    &CB0, &CB1, &CB2, &CA1, &CA2, &CaPML, &CbPML, &CcPML);

  intScheme->template calculateFieldStepInitDiff<grid_type> (&diff11, &diff12, &diff21, &diff22);

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

  CudaGrid<TC> *d_CB0 = NULLPTR;
  CudaGrid<TC> *d_CB1 = NULLPTR;
  CudaGrid<TC> *d_CB2 = NULLPTR;
  CudaGrid<TC> *d_CA1 = NULLPTR;
  CudaGrid<TC> *d_CA2 = NULLPTR;

  CudaGrid<TC> *d_CaPML = NULLPTR;
  CudaGrid<TC> *d_CbPML = NULLPTR;
  CudaGrid<TC> *d_CcPML = NULLPTR;

  SourceCallBack _rightSideFunc = NULLPTR;
  SourceCallBack _borderFunc = NULLPTR;
  SourceCallBack _exactFunc = NULLPTR;

  FPValue _materialModifier;

  if (SOLVER_SETTINGS.getDoUseCuda ()
      && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
  {
    gpuIntSchemeOnGPU->template calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&d_grid, &_gridType,
      &d_materialGrid, &_materialGridType, &d_materialGrid1, &_materialGridType1, &d_materialGrid2, &_materialGridType2,
      &d_materialGrid3, &_materialGridType3, &d_materialGrid4, &_materialGridType4, &d_materialGrid5, &_materialGridType5,
      &d_oppositeGrid1, &d_oppositeGrid2, &d_gridPML1, &_gridPMLType1, &d_gridPML2, &_gridPMLType2,
      &_rightSideFunc, &_borderFunc, &_exactFunc, &_materialModifier, &d_Ca, &d_Cb,
      &d_CB0, &d_CB1, &d_CB2, &d_CA1, &d_CA2, &d_CaPML, &d_CbPML, &d_CcPML);
  }

#endif /* CUDA_ENABLED */

  // TODO: specialize for each dimension
  GridCoordinate3D start3D;
  GridCoordinate3D end3D;

  expandTo3DStartEnd (start, end, start3D, end3D, ct1, ct2, ct3);

  // TODO: remove this check for each iteration
  if (t > 0)
  {
    /*
     * This timestep should be passed to rightside function, which is half step behind grid_type,
     * i.e. exactly on the same time step as opposite fields.
     */
    FPValue timestep;
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        timestep = t;
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        timestep = t;
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        timestep = t;
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        timestep = t + 0.5;
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        timestep = t + 0.5;
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        timestep = t + 0.5;
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <grid_type>
        (d_gpuIntSchemeOnGPU, start3D, end3D,
          timestep, diff11, diff12, diff21, diff22,
          d_grid,
          d_oppositeGrid1,
          d_oppositeGrid2,
          rightSideFunc,
          d_Ca,
          d_Cb,
          usePML,
          gridType, d_materialGrid, materialGridType,
          materialModifier,
          SOLVER_SETTINGS.getDoUseCaCbGrids ());
    }
    else
#endif /* CUDA_ENABLED */
    {
      typename VectorFieldValues<TC>::Iterator iter (start, start, end);
      typename VectorFieldValues<TC>::Iterator iter_end = VectorFieldValues<TC>::Iterator::getEndIterator (start, end);
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

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
          intScheme->template calculateFieldStepIteration<grid_type, true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                             grid, coordFP,
                                                             oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                             usePML,
                                                             gridType, materialGrid, materialGridType,
                                                             materialModifier);
        }
        else
        {
          intScheme->template calculateFieldStepIteration<grid_type, false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                             grid, coordFP,
                                                             oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                             usePML,
                                                             gridType, materialGrid, materialGridType,
                                                             materialModifier);
        }
      }
    }

    bool doComputeCurrentSource = false;
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceJx ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceJy ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceJz ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceMx ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceMy ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        if (SOLVER_SETTINGS.getDoUseCurrentSourceMz ())
        {
          doComputeCurrentSource = true;
        }
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doComputeCurrentSource)
    {
      FieldValue current = FIELDVALUE (0, 0);

#ifdef COMPLEX_FIELD_VALUES
      current = FieldValue (cos (intScheme->getGridTimeStep () * (timestep-1) * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ()),
                             -sin (intScheme->getGridTimeStep () * (timestep-1) * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ()));
#else /* COMPLEX_FIELD_VALUES */
      current = sin (intScheme->getGridTimeStep () * timestep * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ());
#endif /* !COMPLEX_FIELD_VALUES */

      current = current * PhysicsConst::Mu0 / intScheme->getGridTimeStep () / 2.0;

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        // TODO: add cuda call
        ALWAYS_ASSERT (0);
      }
      else
#endif
      {
        if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
        {
          intScheme->template calculateFieldStepIterationCurrent<grid_type, true> (current, grid, Ca, Cb, usePML, gridType, materialGrid, materialGridType, materialModifier);
        }
        else
        {
          intScheme->template calculateFieldStepIterationCurrent<grid_type, false> (current, grid, Ca, Cb, usePML, gridType, materialGrid, materialGridType, materialModifier);
        }
      }
    }

    if (usePML)
    {
      if (useMetamaterials)
      {
#ifdef CUDA_ENABLED
        if (SOLVER_SETTINGS.getDoUseCuda ()
            && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
        {
          gpuIntSchemeOnGPU->calculateFieldStepIterationPMLMetamaterialsKernelLaunch
            (d_gpuIntSchemeOnGPU, start3D, end3D,
             t, d_grid, d_gridPML1,
             d_CB0, d_CB1, d_CB2, d_CA1, d_CA2,
             gridType,
             d_materialGrid1, materialGridType1,
             d_materialGrid2, materialGridType2,
             d_materialGrid3, materialGridType3,
             materialModifier,
             SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ());
        }
        else
#endif /* CUDA_ENABLED */
        {
          typename VectorFieldValues<TC>::Iterator iter (start, start, end);
          typename VectorFieldValues<TC>::Iterator iter_end = VectorFieldValues<TC>::Iterator::getEndIterator (start, end);
          for (; iter != iter_end; ++iter)
          {
            TC pos = iter.getPos ();

            if (SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ())
            {
              intScheme->template calculateFieldStepIterationPMLMetamaterials<true> (t, pos, grid, gridPML1,
                CB0, CB1, CB2, CA1, CA2,
                gridType,
                materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
                materialModifier);
            }
            else
            {
              intScheme->template calculateFieldStepIterationPMLMetamaterials<false> (t, pos, grid, gridPML1,
                CB0, CB1, CB2, CA1, CA2,
                gridType,
                materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
                materialModifier);
            }
          }
        }
      }

#ifdef CUDA_ENABLED
      if (SOLVER_SETTINGS.getDoUseCuda ()
          && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
      {
        gpuIntSchemeOnGPU->template calculateFieldStepIterationPMLKernelLaunch <useMetamaterials> (d_gpuIntSchemeOnGPU, start3D, end3D,
          t, d_grid, d_gridPML1, d_gridPML2, d_CaPML, d_CbPML, d_CcPML, gridPMLType1,
          d_materialGrid1, materialGridType1, d_materialGrid4, materialGridType4, d_materialGrid5, materialGridType5,
          materialModifier,
          SOLVER_SETTINGS.getDoUseCaCbPMLGrids ());
      }
      else
#endif /* CUDA_ENABLED */
      {
        typename VectorFieldValues<TC>::Iterator iter (start, start, end);
        typename VectorFieldValues<TC>::Iterator iter_end = VectorFieldValues<TC>::Iterator::getEndIterator (start, end);
        for (; iter != iter_end; ++iter)
        {
          TC pos = iter.getPos ();

          if (SOLVER_SETTINGS.getDoUseCaCbPMLGrids ())
          {
            intScheme->template calculateFieldStepIterationPML<useMetamaterials, true> (t, pos, grid, gridPML1, gridPML2, CaPML, CbPML, CcPML, gridPMLType1,
              materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
              materialModifier);
          }
          else
          {
            intScheme->template calculateFieldStepIterationPML<useMetamaterials, false> (t, pos, grid, gridPML1, gridPML2, CaPML, CbPML, CcPML, gridPMLType1,
              materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
              materialModifier);
          }
        }
      }
    }
  }

  if (borderFunc != NULLPTR)
  {
    GridCoordinate3D startBorder;
    GridCoordinate3D endBorder;

    TC startB = grid->getTotalSize ().getZero ();
    TC endB = grid->getTotalSize ();

    expandTo3DStartEnd (startB, endB, startBorder, endBorder, ct1, ct2, ct3);

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      gpuIntSchemeOnGPU->template calculateFieldStepIterationBorderKernelLaunch<grid_type>
        (d_gpuIntSchemeOnGPU, startBorder, endBorder, t, d_grid, borderFunc);
    }
    else
#endif
    {
      typename VectorFieldValues<TC>::Iterator iter (startB, startB, endB);
      typename VectorFieldValues<TC>::Iterator iter_end = VectorFieldValues<TC>::Iterator::getEndIterator (startB, endB);
      for (; iter != iter_end; ++iter)
      {
        TC posAbs = iter.getPos ();
        intScheme->template calculateFieldStepIterationBorder<grid_type> (t, posAbs, grid, borderFunc);
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

    GridCoordinate3D startNorm;
    GridCoordinate3D endNorm;

    TC startN = TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);
    TC endN = grid->getTotalSize () - TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

    expandTo3DStartEnd (startN, endN, startNorm, endNorm, ct1, ct2, ct3);

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

    startN = TC::initAxesCoordinate (startNorm.get1 (), startNorm.get2 (), startNorm.get3 (), ct1, ct2, ct3);
    endN = TC::initAxesCoordinate (endNorm.get1 (), endNorm.get2 (), endNorm.get3 (), ct1, ct2, ct3);

    Grid<TC> *normGrid = grid;
    if (usePML)
    {
      normGrid = gridPML2;
    }

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      CudaGrid<TC> *d_normGrid = d_grid;
      if (usePML)
      {
        d_normGrid = d_gridPML2;
      }

      gpuIntSchemeOnGPU->template calculateFieldStepIterationExactKernelLaunch<grid_type>
        (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, startNorm, endNorm, t, d_normGrid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
    }
    else
#endif /* CUDA_ENABLED */
    {
      typename VectorFieldValues<TC>::Iterator iter (startN, startN, endN);
      typename VectorFieldValues<TC>::Iterator iter_end = VectorFieldValues<TC>::Iterator::getEndIterator (startN, endN);
      for (; iter != iter_end; ++iter)
      {
        TC posAbs = iter.getPos ();
        intScheme->template calculateFieldStepIterationExact<grid_type> (t, posAbs, normGrid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
      }
    }

    if (useParallel)
    {
#ifdef PARALLEL_GRID
      FPValue normReTotal = FPValue (0);
      FPValue maxReTotal = FPValue (0);

#ifdef COMPLEX_FIELD_VALUES
      FPValue normImTotal = FPValue (0);
      FPValue normModTotal = FPValue (0);
      FPValue maxImTotal = FPValue (0);
      FPValue maxModTotal = FPValue (0);
#endif

      /*
       * In parallel mode need to share values with all nodes
       */
      for (int process = 0; process < ParallelGrid::getParallelCore ()->getTotalProcCount (); ++process)
      {
        FPValue curRe = FPValue (0);
        FPValue curMaxRe = FPValue (0);

#ifdef COMPLEX_FIELD_VALUES
        FPValue curIm = FPValue (0);
        FPValue curMod = FPValue (0);
        FPValue curMaxIm = FPValue (0);
        FPValue curMaxMod = FPValue (0);
#endif

        if (process == ParallelGrid::getParallelCore ()->getProcessId ())
        {
          curRe = normRe;
          curMaxRe = maxRe;
#ifdef COMPLEX_FIELD_VALUES
          curIm = normIm;
          curMod = normMod;
          curMaxIm = maxIm;
          curMaxMod = maxMod;
#endif
        }

        MPI_Bcast (&curRe, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
        MPI_Bcast (&curMaxRe, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#ifdef COMPLEX_FIELD_VALUES
        MPI_Bcast (&curIm, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
        MPI_Bcast (&curMod, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
        MPI_Bcast (&curMaxIm, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
        MPI_Bcast (&curMaxMod, 1, MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif

        normReTotal += curRe;
        maxReTotal += curMaxRe;
#ifdef COMPLEX_FIELD_VALUES
        normImTotal += curIm;
        normModTotal += curMod;
        maxImTotal += curMaxIm;
        maxModTotal += curMaxMod;
#endif

        MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());
      }

      normRe = normReTotal;
      maxRe = maxReTotal;
#ifdef COMPLEX_FIELD_VALUES
      normIm = normImTotal;
      normMod = normModTotal;
      maxIm = maxImTotal;
      maxMod = maxModTotal;
#endif
#else /* PARALLEL_GRID */
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
    }

#ifdef PARALLEL_GRID
    if (processId == 0)
#endif /* PARALLEL_GRID */
    {
#ifdef COMPLEX_FIELD_VALUES
      normRe = sqrt (normRe / normGrid->getTotalSize ().calculateTotalCoord ());
      normIm = sqrt (normIm / normGrid->getTotalSize ().calculateTotalCoord ());
      normMod = sqrt (normMod / normGrid->getTotalSize ().calculateTotalCoord ());

      /*
       * NOTE: do not change this! test suite depends on the order of values in output
       */
      printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
        normGrid->getName (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
      normRe = sqrt (normRe / normGrid->getTotalSize ().calculateTotalCoord ());

      /*
       * NOTE: do not change this! test suite depends on the order of values in output
       */
      printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
        normGrid->getName (), t, normRe, normRe * 100.0 / maxRe);
#endif
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
uint64_t
Scheme<Type, TCoord, layout_type>::estimateCurrentSize ()
{
  uint64_t size = 0;

  /*
   * Estimation is just size of grid plus size of Grid class
   */

#define GRID_NAME(x, y, steps, time_offset) \
  size += intScheme->has ## x () ? intScheme->get ## x ()->getSize ().calculateTotalCoord () * intScheme->get ## x ()->getCountStoredSteps () * sizeof (FieldValue) + sizeof (Grid<TC>) : 0;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  GRID_NAME(x, y, steps, time_offset)
#include "Grids2.inc.h"
#undef GRID_NAME

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    size += intScheme->getEInc ()->getSize ().calculateTotalCoord () * intScheme->getEInc ()->getCountStoredSteps () * sizeof (FieldValue) + sizeof (Grid<TC>);
    size += intScheme->getHInc ()->getSize ().calculateTotalCoord () * intScheme->getHInc ()->getCountStoredSteps () * sizeof (FieldValue) + sizeof (Grid<TC>);
  }

  /*
   * Add additional 256Mb to cover inaccuracies with small sizes
   */
  size += 256*1024*1024;

  /*
   * multiply on modifier to cover inaccuracies with big sizes
   */
  size *= 1.2;

  return size;
}

#ifdef CUDA_ENABLED
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::setupBlocksForGPU (TC &blockCount, TC &blockSize)
{
  int device = 0;
  cudaCheckErrorCmd (cudaGetDevice (&device));

  cudaDeviceProp prop;
  cudaCheckErrorCmd (cudaGetDeviceProperties (&prop, device));

  uint64_t size = (uint64_t) prop.totalGlobalMem;

  uint64_t requiredSize = estimateCurrentSize ();
  printf ("Estimated current size: %lu byte.\n", requiredSize);

  if (requiredSize < size)
  {
    /*
     * There is enough memory, use single block, which should be already setup in blockCount and blockSize
     */
    return;
  }

  /*
   * Not enough memory, use few blocks
   */

  /*
   * Algorithm is next:
   *
   * Consider AxBxC grid for 3D and AxB for 2D.
   * 1. Split the largest axes until block starts to fit in memory
   *    a) for 2D mode, the border case for this will be 1xB or Ax1, which should certainly fit in memory.
   *       This is true, because otherwise it should not fit in CPU memory. Consider Ax1,
   *       and GPU having 64 Mb of memory (2^26), then, A=2^23, and B should be at least 2^23, which makes
   *       AxB at least 2^46, which is 64 Tb and quite large for RAM (at least for now).
   *    b) for 3D mode, the cases are same to 2D, i.e. 1xBxC, etc. Argumentation is similar to 2D.
   *    c) for 1D mode, it is considered that it always fits.
   *
   * Assert will hit for condition if really huge amount of memory is required, meaning that this case is unsupported yet.
   *
   * Thus, blockCount will be non-equal to 1 for only one axis!
   */

  GridCoordinate3D size3D = expandTo3D (blockSize, ct1, ct2, ct3);
  GridCoordinate3D count3D = expandTo3D (blockCount, ct1, ct2, ct3);

  uint64_t modifier = 1;

  if (size3D.get1 () == size3D.getMax ())
  {
    GridCoordinate3D min3D = GRID_COORDINATE_3D (2, size3D.get2 (), size3D.get3 (), CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    while (size3D >= min3D && requiredSize / modifier >= size)
    {
      size3D.set1 (size3D.get1 () / grid_coord (2));
      count3D.set1 (count3D.get1 () * grid_coord (2));
      modifier *= 2;
      blockSize = TC::initAxesCoordinate (size3D.get1 (), size3D.get2 (), size3D.get3 (), ct1, ct2, ct3);
    }

    if (!(size3D >= min3D))
    {
      ALWAYS_ASSERT_MESSAGE ("Too much memory was requested. Not implemented");
    }
  }
  else if (size3D.get2 () == size3D.getMax ())
  {
    GridCoordinate3D min3D = GRID_COORDINATE_3D (size3D.get1 (), 2, size3D.get3 (), CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    while (size3D >= min3D && requiredSize / modifier >= size)
    {
      size3D.set2 (size3D.get2 () / grid_coord (2));
      count3D.set2 (count3D.get2 () * grid_coord (2));
      modifier *= 2;
      blockSize = TC::initAxesCoordinate (size3D.get1 (), size3D.get2 (), size3D.get3 (), ct1, ct2, ct3);
    }

    if (!(size3D >= min3D))
    {
      ALWAYS_ASSERT_MESSAGE ("Too much memory was requested. Not implemented");
    }
  }
  else if (size3D.get3 () == size3D.getMax ())
  {
    GridCoordinate3D min3D = GRID_COORDINATE_3D (size3D.get1 (), size3D.get2 (), 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    while (size3D >= min3D && requiredSize / modifier >= size)
    {
      size3D.set3 (size3D.get3 () / grid_coord (2));
      count3D.set3 (count3D.get3 () * grid_coord (2));
      modifier *= 2;
      blockSize = TC::initAxesCoordinate (size3D.get1 (), size3D.get2 (), size3D.get3 (), ct1, ct2, ct3);
    }

    if (!(size3D >= min3D))
    {
      ALWAYS_ASSERT_MESSAGE ("Too much memory was requested. Not implemented");
    }
  }
  else
  {
    UNREACHABLE;
  }

  blockCount = TC::initAxesCoordinate (count3D.get1 (), count3D.get2 (), count3D.get3 (), ct1, ct2, ct3);
}
#endif

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initBlocks (time_step t_total)
{
  totalTimeSteps = t_total;

  {
    /*
     * Identify required amount of blocks and their sizes. Left blocks - better.
     */
    blockCount = TC::initAxesCoordinate (1, 1, 1, ct1, ct2, ct3);

#ifdef PARALLEL_GRID
    if (useParallel)
    {
      initParallelBlocks ();
    }
    else
#endif
    {
      blockSize = yeeLayout->getSize ();
    }

#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      setupBlocksForGPU (blockCount, blockSize);
    }
#endif
  }

  printf ("Setup blocks:\n");
  printf ("blockCount:\n");
  blockCount.print ();
  printf ("blockSize:\n");
  blockSize.print ();

  // TODO: remove this check, when correct block setup is implemented
  ALWAYS_ASSERT (blockCount.calculateTotalCoord () == 1);

  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ())
    {
      time_step cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();

      if (blockCount.calculateTotalCoord () > 1
#ifdef PARALLEL_GRID
          || useParallel
#endif
          )
      {
        /*
         * More than one block is used, have to consider buffers now
         */

        /*
         * Buf can't be 1, because this will lead to usage of incorrect data from buffers, as share operations in CUDA
         * mode are delayed until both computations for E and H are finished.
         * That's why additional buffer layer is required for computaions, which basically means, that cudaBuf - 1 time
         * steps can be performed before share operations are required.
         */
        ALWAYS_ASSERT (cudaBuf > 1);
        NTimeSteps = cudaBuf - 1;

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
      }
      else
      {
        NTimeSteps = totalTimeSteps;
        /*
         * Minimum buffer could be used
         */
        ALWAYS_ASSERT (cudaBuf == 1);
      }
    }
    else
#endif /* CUDA_ENABLED */
    {
      /*
       * For non-Cuda builds it's fine to perform single step for block
       */
      NTimeSteps = 1;
    }
  }

#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ()
      && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
  {
    /*
     * Init InternalScheme on GPU
     */
    time_step cudaBuf = 0;

    if (blockCount.calculateTotalCoord () > 1
#ifdef PARALLEL_GRID
        || useParallel
#endif
        )
    {
      cudaBuf = (time_step) SOLVER_SETTINGS.getCudaBlocksBufferSize ();
    }

    gpuIntScheme = new InternalSchemeGPU<Type, TCoord, layout_type> ();
    gpuIntSchemeOnGPU = new InternalSchemeGPU<Type, TCoord, layout_type> ();

    gpuIntScheme->initFromCPU (intScheme, blockSize, TC_COORD (cudaBuf, cudaBuf, cudaBuf, ct1, ct2, ct3));
    gpuIntSchemeOnGPU->initOnGPU (gpuIntScheme);

    cudaCheckErrorCmd (cudaMalloc ((void **) &d_gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>)));
  }
#endif /* CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                           bool parallelLayout,
                                           const TC& totSize,
                                           time_step tStep)
  : useParallel (false)
  , intScheme (new InternalScheme<Type, TCoord, layout_type> ())
#ifdef PARALLEL_GRID
  , eGroup (NULLPTR)
  , hGroup (NULLPTR)
#endif /* PARALLEL_GRID */
  , totalTimeSteps (0)
  , NTimeSteps (0)
#ifdef CUDA_ENABLED
  , gpuIntScheme (NULLPTR)
  , gpuIntSchemeOnGPU (NULLPTR)
  , d_gpuIntSchemeOnGPU (NULLPTR)
#endif /* CUDA_ENABLED */
  , totalEx (NULLPTR)
  , totalEy (NULLPTR)
  , totalEz (NULLPTR)
  , totalHx (NULLPTR)
  , totalHy (NULLPTR)
  , totalHz (NULLPTR)
  , totalInitialized (false)
  , totalEps (NULLPTR)
  , totalMu (NULLPTR)
  , totalOmegaPE (NULLPTR)
  , totalOmegaPM (NULLPTR)
  , totalGammaE (NULLPTR)
  , totalGammaM (NULLPTR)
  , totalStep (tStep)
  , process (-1)
  , numProcs (-1)
  , ct1 (intScheme->get_ct1 ())
  , ct2 (intScheme->get_ct2 ())
  , ct3 (intScheme->get_ct3 ())
  , yeeLayout (layout)
{
  ASSERT (!SOLVER_SETTINGS.getDoUseTFSF ()
          || (SOLVER_SETTINGS.getDoUseTFSF ()
              && (yeeLayout->getLeftBorderTFSF () != TC (0, 0, 0, ct1, ct2, ct3)
                  || yeeLayout->getRightBorderTFSF () != yeeLayout->getSize ())));

  ASSERT (!SOLVER_SETTINGS.getDoUsePML ()
          || (SOLVER_SETTINGS.getDoUsePML () && (yeeLayout->getSizePML () != TC (0, 0, 0, ct1, ct2, ct3))));

  if (SOLVER_SETTINGS.getDoUseMetamaterials () && !SOLVER_SETTINGS.getDoUsePML ())
  {
    ASSERT_MESSAGE ("Metamaterials without pml are not implemented");
  }

#ifdef PARALLEL_GRID
  if (parallelLayout)
  {
    ALWAYS_ASSERT ((TCoord<grid_coord, false>::dimension == ParallelGridCoordinateTemplate<grid_coord, false>::dimension));

    useParallel = true;
  }
#endif

  intScheme->init (layout, useParallel);

  if (!useParallel)
  {
    totalEps = intScheme->getEps ();
    totalMu = intScheme->getMu ();

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      totalOmegaPE = intScheme->getOmegaPE ();
      totalOmegaPM = intScheme->getOmegaPM ();
      totalGammaE = intScheme->getGammaE ();
      totalGammaM = intScheme->getGammaM ();
    }
  }
  else
  {
#ifdef PARALLEL_GRID
    /*
     * We consider two groups here, E and H!
     */
    if (intScheme->getDoNeedEx ())
    {
      eGroup = ((ParallelGrid *) intScheme->getEx ())->getGroup ();
    }
    if (intScheme->getDoNeedEy ())
    {
      ASSERT (eGroup == NULLPTR || eGroup == ((ParallelGrid *) intScheme->getEy ())->getGroup ());
      eGroup = ((ParallelGrid *) intScheme->getEy ())->getGroup ();
    }
    if (intScheme->getDoNeedEz ())
    {
      ASSERT (eGroup == NULLPTR || eGroup == ((ParallelGrid *) intScheme->getEz ())->getGroup ());
      eGroup = ((ParallelGrid *) intScheme->getEz ())->getGroup ();
    }

    ASSERT (eGroup != NULLPTR);

    if (intScheme->getDoNeedHx ())
    {
      hGroup = ((ParallelGrid *) intScheme->getHx ())->getGroup ();
    }
    if (intScheme->getDoNeedHy ())
    {
      ASSERT (hGroup == NULLPTR || hGroup == ((ParallelGrid *) intScheme->getHy ())->getGroup ());
      hGroup = ((ParallelGrid *) intScheme->getHy ())->getGroup ();
    }
    if (intScheme->getDoNeedHz ())
    {
      ASSERT (hGroup == NULLPTR || hGroup == ((ParallelGrid *) intScheme->getHz ())->getGroup ());
      hGroup = ((ParallelGrid *) intScheme->getHz ())->getGroup ();
    }

    ASSERT (hGroup != NULLPTR);
#endif /* PARALLEL_GRID */

    if (SOLVER_SETTINGS.getDoSaveMaterials ())
    {
      totalEps = new Grid<TC> (yeeLayout->getEpsSize (), intScheme->getEps ()->getCountStoredSteps (), "Eps");
      totalMu = new Grid<TC> (yeeLayout->getMuSize (), intScheme->getMu ()->getCountStoredSteps (), "Mu");

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        totalOmegaPE = new Grid<TC> (yeeLayout->getEpsSize (), intScheme->getOmegaPE ()->getCountStoredSteps (), "OmegaPE");
        totalOmegaPM = new Grid<TC> (yeeLayout->getEpsSize (), intScheme->getOmegaPM ()->getCountStoredSteps (), "OmegaPM");
        totalGammaE = new Grid<TC> (yeeLayout->getEpsSize (), intScheme->getGammaE ()->getCountStoredSteps (), "GammaE");
        totalGammaM = new Grid<TC> (yeeLayout->getEpsSize (), intScheme->getGammaM ()->getCountStoredSteps (), "GammaM");
      }
    }
  }

  if (SOLVER_SETTINGS.getDoSaveAsBMP ())
  {
    PaletteType palette = PaletteType::PALETTE_GRAY;
    OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

    if (SOLVER_SETTINGS.getDoUsePaletteGray ())
    {
      palette = PaletteType::PALETTE_GRAY;
    }
    else if (SOLVER_SETTINGS.getDoUsePaletteRGB ())
    {
      palette = PaletteType::PALETTE_BLUE_GREEN_RED;
    }

    if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
    {
      orthogonalAxis = OrthogonalAxis::X;
    }
    else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
    {
      orthogonalAxis = OrthogonalAxis::Y;
    }
    else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
    {
      orthogonalAxis = OrthogonalAxis::Z;
    }

    dumper[FILE_TYPE_BMP] = new BMPDumper<TC> ();
    ((BMPDumper<TC> *) dumper[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);

    dumper1D[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate1D> ();
    ((BMPDumper<GridCoordinate1D> *) dumper1D[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  else
  {
    dumper[FILE_TYPE_BMP] = NULLPTR;
    dumper1D[FILE_TYPE_BMP] = NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoSaveAsDAT ())
  {
    dumper[FILE_TYPE_DAT] = new DATDumper<TC> ();
    dumper1D[FILE_TYPE_DAT] = new DATDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_DAT] = NULLPTR;
    dumper1D[FILE_TYPE_DAT] = NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoSaveAsTXT ())
  {
    dumper[FILE_TYPE_TXT] = new TXTDumper<TC> ();
    dumper1D[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_TXT] = NULLPTR;
    dumper1D[FILE_TYPE_TXT] = NULLPTR;
  }

  if (!SOLVER_SETTINGS.getEpsFileName ().empty ()
      || !SOLVER_SETTINGS.getMuFileName ().empty ()
      || !SOLVER_SETTINGS.getOmegaPEFileName ().empty ()
      || !SOLVER_SETTINGS.getOmegaPMFileName ().empty ()
      || !SOLVER_SETTINGS.getGammaEFileName ().empty ()
      || !SOLVER_SETTINGS.getGammaMFileName ().empty ())
  {
    {
      loader[FILE_TYPE_BMP] = new BMPLoader<TC> ();

      PaletteType palette = PaletteType::PALETTE_GRAY;
      OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

      if (SOLVER_SETTINGS.getDoUsePaletteGray ())
      {
        palette = PaletteType::PALETTE_GRAY;
      }
      else if (SOLVER_SETTINGS.getDoUsePaletteRGB ())
      {
        palette = PaletteType::PALETTE_BLUE_GREEN_RED;
      }

      if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
      {
        orthogonalAxis = OrthogonalAxis::X;
      }
      else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
      {
        orthogonalAxis = OrthogonalAxis::Y;
      }
      else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
      {
        orthogonalAxis = OrthogonalAxis::Z;
      }

      ((BMPLoader<TC> *) loader[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
    }
    {
      loader[FILE_TYPE_DAT] = new DATLoader<TC> ();
    }
    {
      loader[FILE_TYPE_TXT] = new TXTLoader<TC> ();
    }
  }
  else
  {
    loader[FILE_TYPE_BMP] = NULLPTR;
    loader[FILE_TYPE_DAT] = NULLPTR;
    loader[FILE_TYPE_TXT] = NULLPTR;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::~Scheme ()
{
#ifdef CUDA_ENABLED
  if (SOLVER_SETTINGS.getDoUseCuda ()
      && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
  {
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

  if (totalInitialized)
  {
    delete totalEx;
    delete totalEy;
    delete totalEz;

    delete totalHx;
    delete totalHy;
    delete totalHz;
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    delete totalEps;
    delete totalMu;

    delete totalOmegaPE;
    delete totalOmegaPM;
    delete totalGammaE;
    delete totalGammaM;
#else /* PARALLEL_GRID */
    UNREACHABLE;
#endif /* !PARALLEL_GRID */
  }

  delete dumper[FILE_TYPE_BMP];
  delete dumper[FILE_TYPE_DAT];
  delete dumper[FILE_TYPE_TXT];

  delete loader[FILE_TYPE_BMP];
  delete loader[FILE_TYPE_DAT];
  delete loader[FILE_TYPE_TXT];

  delete dumper1D[FILE_TYPE_BMP];
  delete dumper1D[FILE_TYPE_DAT];
  delete dumper1D[FILE_TYPE_TXT];
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initScheme (FPValue dx, FPValue sourceWaveLen, time_step t_total)
{
  intScheme->initScheme (dx, sourceWaveLen);

  initCallBacks ();
  initGrids ();
  initBlocks (t_total);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initCallBacks ()
{
  /*
   * NOTE: with CUDA enabled, CPU internal scheme will contain GPU callbacks!
   *       except for initial callbacks
   */
#ifndef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUsePolinom1BorderCondition ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyBorder (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzBorder (CallBack::polinom1_ez);
      intScheme->setCallbackHyBorder (CallBack::polinom1_hy);
    }
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2BorderCondition ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ex, sizeof(SourceCallBack)));
      intScheme->setCallbackExBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ey, sizeof(SourceCallBack)));
      intScheme->setCallbackEyBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzBorder (tmp);

      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzBorder (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExBorder (CallBack::polinom2_ex);
      intScheme->setCallbackEyBorder (CallBack::polinom2_ey);
      intScheme->setCallbackEzBorder (CallBack::polinom2_ez);

      intScheme->setCallbackHxBorder (CallBack::polinom2_hx);
      intScheme->setCallbackHyBorder (CallBack::polinom2_hy);
      intScheme->setCallbackHzBorder (CallBack::polinom2_hz);
    }
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3BorderCondition ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyBorder (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzBorder (CallBack::polinom3_ez);
      intScheme->setCallbackHyBorder (CallBack::polinom3_hy);
    }
  }
  else if (SOLVER_SETTINGS.getDoUseSin1BorderCondition ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_sin1_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzBorder (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_sin1_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyBorder (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzBorder (CallBack::sin1_ez);
      intScheme->setCallbackHyBorder (CallBack::sin1_hy);
    }
  }

  if (SOLVER_SETTINGS.getDoUsePolinom1StartValues ())
  {
    intScheme->setCallbackEzInitial (CallBack::polinom1_ez);
    intScheme->setCallbackHyInitial (CallBack::polinom1_hy);
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2StartValues ())
  {
    intScheme->setCallbackExInitial (CallBack::polinom2_ex);
    intScheme->setCallbackEyInitial (CallBack::polinom2_ey);
    intScheme->setCallbackEzInitial (CallBack::polinom2_ez);

    intScheme->setCallbackHxInitial (CallBack::polinom2_hx);
    intScheme->setCallbackHyInitial (CallBack::polinom2_hy);
    intScheme->setCallbackHzInitial (CallBack::polinom2_hz);
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3StartValues ())
  {
    intScheme->setCallbackEzInitial (CallBack::polinom3_ez);
    intScheme->setCallbackHyInitial (CallBack::polinom3_hy);
  }
  else if (SOLVER_SETTINGS.getDoUseSin1StartValues ())
  {
    intScheme->setCallbackEzInitial (CallBack::sin1_ez);
    intScheme->setCallbackHyInitial (CallBack::sin1_hy);
  }

  if (SOLVER_SETTINGS.getDoUsePolinom1RightSide ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_jz, sizeof(SourceCallBack)));
      intScheme->setCallbackJz (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_my, sizeof(SourceCallBack)));
      intScheme->setCallbackMy (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackJz (CallBack::polinom1_jz);
      intScheme->setCallbackMy (CallBack::polinom1_my);
    }
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2RightSide ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_jx, sizeof(SourceCallBack)));
      intScheme->setCallbackJx (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_jy, sizeof(SourceCallBack)));
      intScheme->setCallbackJy (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_jz, sizeof(SourceCallBack)));
      intScheme->setCallbackJz (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_mx, sizeof(SourceCallBack)));
      intScheme->setCallbackMx (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_my, sizeof(SourceCallBack)));
      intScheme->setCallbackMy (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_mz, sizeof(SourceCallBack)));
      intScheme->setCallbackMz (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackJx (CallBack::polinom2_jx);
      intScheme->setCallbackJy (CallBack::polinom2_jy);
      intScheme->setCallbackJz (CallBack::polinom2_jz);

      intScheme->setCallbackMx (CallBack::polinom2_mx);
      intScheme->setCallbackMy (CallBack::polinom2_my);
      intScheme->setCallbackMz (CallBack::polinom2_mz);
    }
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3RightSide ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_jz, sizeof(SourceCallBack)));
      intScheme->setCallbackJz (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_my, sizeof(SourceCallBack)));
      intScheme->setCallbackMy (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackJz (CallBack::polinom3_jz);
      intScheme->setCallbackMy (CallBack::polinom3_my);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculatePolinom1DiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom1_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::polinom1_ez);
      intScheme->setCallbackHyExact (CallBack::polinom1_hy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom2DiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ex, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ey, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom2_hz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::polinom2_ex);
      intScheme->setCallbackEyExact (CallBack::polinom2_ey);
      intScheme->setCallbackEzExact (CallBack::polinom2_ez);

      intScheme->setCallbackHxExact (CallBack::polinom2_hx);
      intScheme->setCallbackHyExact (CallBack::polinom2_hy);
      intScheme->setCallbackHzExact (CallBack::polinom2_hz);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom3DiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_polinom3_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::polinom3_ez);
      intScheme->setCallbackHyExact (CallBack::polinom3_hy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateSin1DiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_sin1_ez, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_sin1_hy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::sin1_ez);
      intScheme->setCallbackHyExact (CallBack::sin1_hy);
    }
  }
#endif

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ex_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hy_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp1_ex_exhy);
      intScheme->setCallbackHyExact (CallBack::exp1_hy_exhy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ex_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hy_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp2_ex_exhy);
      intScheme->setCallbackHyExact (CallBack::exp2_hy_exhy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ex_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hy_exhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp3_ex_exhy);
      intScheme->setCallbackHyExact (CallBack::exp3_hy_exhy);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ex_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hz_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp1_ex_exhz);
      intScheme->setCallbackHzExact (CallBack::exp1_hz_exhz);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ex_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hz_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp2_ex_exhz);
      intScheme->setCallbackHzExact (CallBack::exp2_hz_exhz);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ex_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackExExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hz_exhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackExExact (CallBack::exp3_ex_exhz);
      intScheme->setCallbackHzExact (CallBack::exp3_hz_exhz);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ey_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hx_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp1_ey_eyhx);
      intScheme->setCallbackHxExact (CallBack::exp1_hx_eyhx);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ey_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hx_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp2_ey_eyhx);
      intScheme->setCallbackHxExact (CallBack::exp2_hx_eyhx);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ey_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hx_eyhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp3_ey_eyhx);
      intScheme->setCallbackHxExact (CallBack::exp3_hx_eyhx);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ey_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hz_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp1_ey_eyhz);
      intScheme->setCallbackHzExact (CallBack::exp1_hz_eyhz);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ey_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hz_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp2_ey_eyhz);
      intScheme->setCallbackHzExact (CallBack::exp2_hz_eyhz);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ey_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackEyExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hz_eyhz, sizeof(SourceCallBack)));
      intScheme->setCallbackHzExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEyExact (CallBack::exp3_ey_eyhz);
      intScheme->setCallbackHzExact (CallBack::exp3_hz_eyhz);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ez_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hx_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp1_ez_ezhx);
      intScheme->setCallbackHxExact (CallBack::exp1_hx_ezhx);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ez_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hx_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp2_ez_ezhx);
      intScheme->setCallbackHxExact (CallBack::exp2_hx_ezhx);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ez_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hx_ezhx, sizeof(SourceCallBack)));
      intScheme->setCallbackHxExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp3_ez_ezhx);
      intScheme->setCallbackHxExact (CallBack::exp3_hx_ezhx);
    }
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_ez_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp1_hy_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp1_ez_ezhy);
      intScheme->setCallbackHyExact (CallBack::exp1_hy_ezhy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_ez_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp2_hy_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp2_ez_ezhy);
      intScheme->setCallbackHyExact (CallBack::exp2_hy_ezhy);
    }
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ()
        && SOLVER_SETTINGS.getIndexOfGPUForCurrentNode () != NO_GPU)
    {
      SourceCallBack tmp;
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_ez_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackEzExact (tmp);
      cudaCheckErrorCmd (cudaMemcpyFromSymbol (&tmp, d_exp3_hy_ezhy, sizeof(SourceCallBack)));
      intScheme->setCallbackHyExact (tmp);
    }
    else
#endif
    {
      intScheme->setCallbackEzExact (CallBack::exp3_ez_ezhy);
      intScheme->setCallbackHyExact (CallBack::exp3_hy_ezhy);
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initMaterialFromFile (GridType gridType, Grid<TC> *grid, Grid<TC> *totalGrid)
{
  std::string filename;

  switch (gridType)
  {
    case GridType::EPS:
    {
      filename = SOLVER_SETTINGS.getEpsFileName ();
      break;
    }
    case GridType::MU:
    {
      filename = SOLVER_SETTINGS.getMuFileName ();
      break;
    }
    case GridType::OMEGAPE:
    {
      filename = SOLVER_SETTINGS.getOmegaPEFileName ();
      break;
    }
    case GridType::OMEGAPM:
    {
      filename = SOLVER_SETTINGS.getOmegaPMFileName ();
      break;
    }
    case GridType::GAMMAE:
    {
      filename = SOLVER_SETTINGS.getGammaEFileName ();
      break;
    }
    case GridType::GAMMAM:
    {
      filename = SOLVER_SETTINGS.getGammaMFileName ();
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (filename.empty ())
  {
    return;
  }

  TC zero = TC_COORD (0, 0, 0, ct1, ct2, ct3);

  FileType type = GridFileManager::getFileType (filename);

  std::vector< std::string > fileNames (1);
  fileNames[0] = filename;

  loader[type]->loadGrid (totalGrid, zero, totalGrid->getSize (), 0, 0, fileNames);

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    typename VectorFieldValues<TC>::Iterator iter = grid->begin ();
    typename VectorFieldValues<TC>::Iterator iter_end = grid->end ();
    for (; iter != iter_end; ++iter)
    {
      TC pos = iter.getPos ();
      TC posAbs = grid->getTotalPosition (pos);

      FieldValue *val = grid->getFieldValue (pos, 0);
      *val = *totalGrid->getFieldValue (posAbs, 0);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGridWithInitialVals (GridType gridType, Grid<TC> *grid, FPValue timestep)
{
  SourceCallBack cb = NULLPTR;

  switch (gridType)
  {
    case GridType::EX:
    {
      cb = intScheme->getCallbackExInitial ();
      break;
    }
    case GridType::EY:
    {
      cb = intScheme->getCallbackEyInitial ();
      break;
    }
    case GridType::EZ:
    {
      cb = intScheme->getCallbackEzInitial ();
      break;
    }
    case GridType::HX:
    {
      cb = intScheme->getCallbackHxInitial ();
      break;
    }
    case GridType::HY:
    {
      cb = intScheme->getCallbackHyInitial ();
      break;
    }
    case GridType::HZ:
    {
      cb = intScheme->getCallbackHzInitial ();
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (cb == NULLPTR)
  {
    return;
  }

  typename VectorFieldValues<TC>::Iterator iter = grid->begin ();
  typename VectorFieldValues<TC>::Iterator iter_end = grid->end ();
  for (; iter != iter_end; ++iter)
  {
    TC pos = iter.getPos ();
    TC posAbs = grid->getTotalPosition (pos);
    TCFP realCoord;

    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    grid->setFieldValue (cb (expandTo3D (realCoord * intScheme->getGridStep (), ct1, ct2, ct3), timestep), pos, 0);
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGrids ()
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  intScheme->getEps ()->initialize (FieldValueHelpers::getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::EPS, intScheme->getEps (), totalEps);

  if (SOLVER_SETTINGS.getEpsSphere () != 1)
  {
    typename VectorFieldValues<TC>::Iterator iter = intScheme->getEps ()->begin ();
    typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEps ()->end ();
    for (; iter != iter_end; ++iter)
    {
      TC pos = iter.getPos ();
      TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getEps ()->getTotalPosition (pos));
      FieldValue *val = intScheme->getEps ()->getFieldValue (pos, 0);

      FieldValue epsVal = FieldValueHelpers::getFieldValueRealOnly (SOLVER_SETTINGS.getEpsSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getEpsSphereCenterX (),
                                             SOLVER_SETTINGS.getEpsSphereCenterY (),
                                             SOLVER_SETTINGS.getEpsSphereCenterZ (),
                                             ct1, ct2, ct3);
      if (SOLVER_SETTINGS.getDoUseStairApproximation ())
      {
        *val = Approximation::approximateSphereStair (posAbs,
                                                         center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                         SOLVER_SETTINGS.getEpsSphereRadius () * modifier,
                                                         epsVal,
                                                         FieldValueHelpers::getFieldValueRealOnly (1.0));
      }
      else
      {
        *val = Approximation::approximateSphereAccurate (posAbs,
                                                         center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                         SOLVER_SETTINGS.getEpsSphereRadius () * modifier,
                                                         epsVal,
                                                         FieldValueHelpers::getFieldValueRealOnly (1.0));
      }
    }
  }
  if (SOLVER_SETTINGS.getUseEpsAllNorm ())
  {
    typename VectorFieldValues<TC>::Iterator iter = intScheme->getEps ()->begin ();
    for (; iter != intScheme->getEps ()->end (); ++iter)
    {
      TC pos = iter.getPos ();
      FieldValue *val = intScheme->getEps ()->getFieldValue (pos, 0);
      *val = FieldValueHelpers::getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Eps0);
    }
  }

  intScheme->getMu ()->initialize (FieldValueHelpers::getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::MU, intScheme->getMu (), totalMu);

  if (SOLVER_SETTINGS.getMuSphere () != 1)
  {
    typename VectorFieldValues<TC>::Iterator iter = intScheme->getMu ()->begin ();
    typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getMu ()->end ();
    for (; iter != iter_end; ++iter)
    {
      TC pos = iter.getPos ();
      TCFP posAbs = yeeLayout->getMuCoordFP (intScheme->getMu ()->getTotalPosition (pos));
      FieldValue *val = intScheme->getMu ()->getFieldValue (pos, 0);

      FieldValue muVal = FieldValueHelpers::getFieldValueRealOnly (SOLVER_SETTINGS.getMuSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getMuSphereCenterX (),
                                             SOLVER_SETTINGS.getMuSphereCenterY (),
                                             SOLVER_SETTINGS.getMuSphereCenterZ (),
                                             ct1, ct2, ct3);
      if (SOLVER_SETTINGS.getDoUseStairApproximation ())
      {
        *val = Approximation::approximateSphereStair (posAbs,
                                                         center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                         SOLVER_SETTINGS.getMuSphereRadius () * modifier,
                                                         muVal,
                                                         FieldValueHelpers::getFieldValueRealOnly (1.0));
      }
      else
      {
        *val = Approximation::approximateSphereAccurate (posAbs,
                                                         center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                         SOLVER_SETTINGS.getMuSphereRadius () * modifier,
                                                         muVal,
                                                         FieldValueHelpers::getFieldValueRealOnly (1.0));
      }
    }
  }
  if (SOLVER_SETTINGS.getUseMuAllNorm ())
  {
    typename VectorFieldValues<TC>::Iterator iter = intScheme->getMu ()->begin ();
    for (; iter != intScheme->getMu ()->end (); ++iter)
    {
      TC pos = iter.getPos ();
      FieldValue *val = intScheme->getMu ()->getFieldValue (pos, 0);
      *val = FieldValueHelpers::getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Mu0);
    }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    initMaterialFromFile (GridType::OMEGAPE, intScheme->getOmegaPE (), totalOmegaPE);

    if (SOLVER_SETTINGS.getOmegaPESphere () != 0)
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getOmegaPE ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getOmegaPE ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();
        TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getOmegaPE ()->getTotalPosition (pos));
        FieldValue *val = intScheme->getOmegaPE ()->getFieldValue (pos, 0);

        FieldValue omegapeVal = FieldValueHelpers::getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPESphere () * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ());

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPESphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterZ (),
                                                ct1, ct2, ct3);
        if (SOLVER_SETTINGS.getDoUseStairApproximation ())
        {
          *val = Approximation::approximateSphereStair (posAbs,
                                                                    center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                                    SOLVER_SETTINGS.getOmegaPESphereRadius () * modifier,
                                                                    omegapeVal,
                                                                    FieldValueHelpers::getFieldValueRealOnly (0.0));
        }
        else
        {
          *val = Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                                    SOLVER_SETTINGS.getOmegaPESphereRadius () * modifier,
                                                                    omegapeVal,
                                                                    FieldValueHelpers::getFieldValueRealOnly (0.0));
        }
      }
    }

    initMaterialFromFile (GridType::OMEGAPM, intScheme->getOmegaPM (), totalOmegaPM);

    if (SOLVER_SETTINGS.getOmegaPMSphere () != 0)
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getOmegaPM ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getOmegaPM ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();
        TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getOmegaPM ()->getTotalPosition (pos));
        FieldValue *val = intScheme->getOmegaPM ()->getFieldValue (pos, 0);

        FieldValue omegapmVal = FieldValueHelpers::getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPMSphere () * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ());

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPMSphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterZ (),
                                                ct1, ct2, ct3);
        if (SOLVER_SETTINGS.getDoUseStairApproximation ())
        {
          *val = Approximation::approximateSphereStair (posAbs,
                                                                    center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                                    SOLVER_SETTINGS.getOmegaPMSphereRadius () * modifier,
                                                                    omegapmVal,
                                                                    FieldValueHelpers::getFieldValueRealOnly (0.0));
        }
        else
        {
          *val = Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                                    SOLVER_SETTINGS.getOmegaPMSphereRadius () * modifier,
                                                                    omegapmVal,
                                                                    FieldValueHelpers::getFieldValueRealOnly (0.0));
        }
      }
    }

    initMaterialFromFile (GridType::GAMMAE, intScheme->getGammaE (), totalGammaE);

    initMaterialFromFile (GridType::GAMMAM, intScheme->getGammaM (), totalGammaM);
  }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    initSigmas ();
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (SOLVER_SETTINGS.getDoSaveMaterials ())
    {
      if (useParallel)
      {
        initFullMaterialGrids ();
      }

      if (processId == 0)
      {
        TC startEps, startMu, startOmegaPE, startOmegaPM, startGammaE, startGammaM;
        TC endEps, endMu, endOmegaPE, endOmegaPM, endGammaE, endGammaM;

        if (SOLVER_SETTINGS.getDoUseManualStartEndDumpCoord ())
        {
          TC start = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveStartCoordX (),
                                            SOLVER_SETTINGS.getSaveStartCoordY (),
                                            SOLVER_SETTINGS.getSaveStartCoordZ (),
                                            ct1, ct2, ct3);
          TC end = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveEndCoordX (),
                                          SOLVER_SETTINGS.getSaveEndCoordY (),
                                          SOLVER_SETTINGS.getSaveEndCoordZ (),
                                          ct1, ct2, ct3);
          startEps = startMu = startOmegaPE = startOmegaPM = startGammaE = startGammaM = start;
          endEps = endMu = endOmegaPE = endOmegaPM = endGammaE = endGammaM = end;
        }
        else
        {
          startEps = getStartCoord (GridType::EPS, totalEps->getSize ());
          endEps = getEndCoord (GridType::EPS, totalEps->getSize ());

          startMu = getStartCoord (GridType::MU, totalMu->getSize ());
          endMu = getEndCoord (GridType::MU, totalMu->getSize ());

          if (SOLVER_SETTINGS.getDoUseMetamaterials ())
          {
            startOmegaPE = getStartCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());
            endOmegaPE = getEndCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());

            startOmegaPM = getStartCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());
            endOmegaPM = getEndCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());

            startGammaE = getStartCoord (GridType::GAMMAE, totalGammaE->getSize ());
            endGammaE = getEndCoord (GridType::GAMMAE, totalGammaE->getSize ());

            startGammaM = getStartCoord (GridType::GAMMAM, totalGammaM->getSize ());
            endGammaM = getEndCoord (GridType::GAMMAM, totalGammaM->getSize ());
          }
        }

        dumper[type]->dumpGrid (totalEps, startEps, endEps, 0, 0, processId);
        dumper[type]->dumpGrid (totalMu, startMu, endMu, 0, 0, processId);

        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          dumper[type]->dumpGrid (totalOmegaPE, startOmegaPE, endOmegaPE, 0, 0, processId);
          dumper[type]->dumpGrid (totalOmegaPM, startOmegaPM, endOmegaPM, 0, 0, processId);
          dumper[type]->dumpGrid (totalGammaE, startGammaE, endGammaE, 0, 0, processId);
          dumper[type]->dumpGrid (totalGammaM, startGammaM, endGammaM, 0, 0, processId);
        }
        //
        // if (SOLVER_SETTINGS.getDoUsePML ())
        // {
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaX");
        //   dumper[type]->dumpGrid (internalScheme.SigmaX,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaX->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaX->getSize ().get1 (), internalScheme.SigmaX->getSize ().get2 (), internalScheme.SigmaX->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaY");
        //   dumper[type]->dumpGrid (internalScheme.SigmaY,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaY->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaY->getSize ().get1 (), internalScheme.SigmaY->getSize ().get2 (), internalScheme.SigmaY->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaZ");
        //   dumper[type]->dumpGrid (internalScheme.SigmaZ,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaZ->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaZ->getSize ().get1 (), internalScheme.SigmaZ->getSize ().get2 (), internalScheme.SigmaZ->getSize ().get3 () / 2 + 1));
        // }
      }
    }
  }

  if (intScheme->getDoNeedEx ())
  {
    initGridWithInitialVals (GridType::EX, intScheme->getEx (), 0.5 * intScheme->getGridTimeStep ());
  }
  if (intScheme->getDoNeedEy ())
  {
    initGridWithInitialVals (GridType::EY, intScheme->getEy (), 0.5 * intScheme->getGridTimeStep ());
  }
  if (intScheme->getDoNeedEz ())
  {
    initGridWithInitialVals (GridType::EZ, intScheme->getEz (), 0.5 * intScheme->getGridTimeStep ());
  }

  if (intScheme->getDoNeedHx ())
  {
    initGridWithInitialVals (GridType::HX, intScheme->getHx (), intScheme->getGridTimeStep ());
  }
  if (intScheme->getDoNeedHy ())
  {
    initGridWithInitialVals (GridType::HY, intScheme->getHy (), intScheme->getGridTimeStep ());
  }
  if (intScheme->getDoNeedHz ())
  {
    initGridWithInitialVals (GridType::HZ, intScheme->getHz (), intScheme->getGridTimeStep ());
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());

    ((ParallelGrid *) intScheme->getEps ())->share ();
    ((ParallelGrid *) intScheme->getMu ())->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (intScheme->getDoNeedSigmaX ())
      {
        ((ParallelGrid *) intScheme->getSigmaX ())->share ();
      }
      if (intScheme->getDoNeedSigmaY ())
      {
        ((ParallelGrid *) intScheme->getSigmaY ())->share ();
      }
      if (intScheme->getDoNeedSigmaZ ())
      {
        ((ParallelGrid *) intScheme->getSigmaZ ())->share ();
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
  {
    if (intScheme->getDoNeedEx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getExStartDiff () && pos < intScheme->getEx ()->getSize () - yeeLayout->getExEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getEx ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::EX, intScheme->getSigmaY (), GridType::SIGMAY) : 0;
          FPValue dd = (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ()) / dd;
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ()) / dd;
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::EX, intScheme->getEps (), GridType::EPS);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * intScheme->getGridStep ());
        }

        intScheme->getCaEx ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbEx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEyStartDiff () && pos < intScheme->getEy ()->getSize () - yeeLayout->getEyEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getEy ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::EY, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;
          FPValue dd = (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ()) / dd;
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ()) / dd;
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::EY, intScheme->getEps (), GridType::EPS);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * intScheme->getGridStep ());
        }

        intScheme->getCaEy ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbEy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEzStartDiff () && pos < intScheme->getEz ()->getSize () - yeeLayout->getEzEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getEz ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::EZ, intScheme->getSigmaX (), GridType::SIGMAX) : 0;
          FPValue dd = (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ()) / dd;
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ()) / dd;
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::EZ, intScheme->getEps (), GridType::EPS);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Eps0 * intScheme->getGridStep ());
        }

        intScheme->getCaEz ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbEz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHxStartDiff () && pos < intScheme->getHx ()->getSize () - yeeLayout->getHxEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getHx ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::HX, intScheme->getSigmaY (), GridType::SIGMAY) : 0;
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::HX, intScheme->getMu (), GridType::MU);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * intScheme->getGridStep ());
        }

        intScheme->getDaHx ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbHx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHyStartDiff () && pos < intScheme->getHy ()->getSize () - yeeLayout->getHyEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getHy ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::HY, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::HY, intScheme->getMu (), GridType::MU);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * intScheme->getGridStep ());
        }

        intScheme->getDaHy ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbHy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHzStartDiff () && pos < intScheme->getHz ()->getSize () - yeeLayout->getHzEndDiff ()))
        {
          continue;
        }

        FPValue Ca;
        FPValue Cb;

        FPValue k_mod = FPValue (1);

        TC posAbs = intScheme->getHz ()->getTotalPosition (pos);

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          FPValue material = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::HZ, intScheme->getSigmaX (), GridType::SIGMAX) : 0;
          Ca = (2 * PhysicsConst::Eps0 * k_mod - material * intScheme->getGridTimeStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
          Cb = (2 * PhysicsConst::Eps0 * intScheme->getGridTimeStep () / intScheme->getGridStep ())
               / (2 * PhysicsConst::Eps0 * k_mod + material * intScheme->getGridTimeStep ());
        }
        else
        {
          FPValue material = intScheme->getMaterial (posAbs, GridType::HZ, intScheme->getMu (), GridType::MU);
          Ca = FPValue (1);
          Cb = intScheme->getGridTimeStep () / (material * PhysicsConst::Mu0 * intScheme->getGridStep ());
        }

        intScheme->getDaHz ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbHz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
  }

  if (SOLVER_SETTINGS.getDoUseCaCbPMLGrids () && SOLVER_SETTINGS.getDoUsePML ())
  {
    if (intScheme->getDoNeedEx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getExStartDiff () && pos < intScheme->getEx ()->getSize () - yeeLayout->getExEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getEx ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::DX, intScheme->getEps (), GridType::EPS);
        FPValue material4 = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::DX, intScheme->getSigmaX (), GridType::SIGMAX) : 0;
        FPValue material5 = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::DX, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;

        FPValue modifier = material1 * PhysicsConst::Eps0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getCaPMLEx ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbPMLEx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getCcPMLEx ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEyStartDiff () && pos < intScheme->getEy ()->getSize () - yeeLayout->getEyEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getEy ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::DY, intScheme->getEps (), GridType::EPS);
        FPValue material4 = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::DY, intScheme->getSigmaY (), GridType::SIGMAY) : 0;
        FPValue material5 = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::DY, intScheme->getSigmaX (), GridType::SIGMAX) : 0;

        FPValue modifier = material1 * PhysicsConst::Eps0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getCaPMLEy ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbPMLEy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getCcPMLEy ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEzStartDiff () && pos < intScheme->getEz ()->getSize () - yeeLayout->getEzEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getEz ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::DZ, intScheme->getEps (), GridType::EPS);
        FPValue material4 = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::DZ, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;
        FPValue material5 = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::DZ, intScheme->getSigmaY (), GridType::SIGMAY) : 0;

        FPValue modifier = material1 * PhysicsConst::Eps0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getCaPMLEz ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getCbPMLEz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getCcPMLEz ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHxStartDiff () && pos < intScheme->getHx ()->getSize () - yeeLayout->getHxEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getHx ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::BX, intScheme->getMu (), GridType::MU);
        FPValue material4 = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::BX, intScheme->getSigmaX (), GridType::SIGMAX) : 0;
        FPValue material5 = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::BX, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;

        FPValue modifier = material1 * PhysicsConst::Mu0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getDaPMLHx ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbPMLHx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getDcPMLHx ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHyStartDiff () && pos < intScheme->getHy ()->getSize () - yeeLayout->getHyEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getHy ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::BY, intScheme->getMu (), GridType::MU);
        FPValue material4 = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::BY, intScheme->getSigmaY (), GridType::SIGMAY) : 0;
        FPValue material5 = intScheme->hasSigmaX () ? intScheme->getMaterial (posAbs, GridType::BY, intScheme->getSigmaX (), GridType::SIGMAX) : 0;

        FPValue modifier = material1 * PhysicsConst::Mu0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getDaPMLHy ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbPMLHy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getDcPMLHy ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHzStartDiff () && pos < intScheme->getHz ()->getSize () - yeeLayout->getHzEndDiff ()))
        {
          continue;
        }

        FPValue k_mod1 = FPValue (1);
        FPValue k_mod2 = FPValue (1);

        TC posAbs = intScheme->getHz ()->getTotalPosition (pos);

        FPValue material1 = intScheme->getMaterial (posAbs, GridType::BZ, intScheme->getMu (), GridType::MU);
        FPValue material4 = intScheme->hasSigmaZ () ? intScheme->getMaterial (posAbs, GridType::BZ, intScheme->getSigmaZ (), GridType::SIGMAZ) : 0;
        FPValue material5 = intScheme->hasSigmaY () ? intScheme->getMaterial (posAbs, GridType::BZ, intScheme->getSigmaY (), GridType::SIGMAY) : 0;

        FPValue modifier = material1 * PhysicsConst::Mu0;
        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          modifier = FPValue (1);
        }

        FPValue eps0 = PhysicsConst::Eps0;
        FPValue dd = (2 * eps0 * k_mod2 + material5 * intScheme->getGridTimeStep ());
        FPValue Ca = (2 * eps0 * k_mod2 - material5 * intScheme->getGridTimeStep ()) / dd;
        FPValue Cb = ((2 * eps0 * k_mod1 + material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;
        FPValue Cc = ((2 * eps0 * k_mod1 - material4 * intScheme->getGridTimeStep ()) / (modifier)) / dd;

        intScheme->getDaPMLHz ()->setFieldValue (FIELDVALUE (Ca, FPValue (0)), pos, 0);
        intScheme->getDbPMLHz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
        intScheme->getDcPMLHz ()->setFieldValue (FIELDVALUE (Cc, FPValue (0)), pos, 0);
      }
    }
  }

  if (SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ()
      && SOLVER_SETTINGS.getDoUsePML ()
      && SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    if (intScheme->getDoNeedEx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getExStartDiff () && pos < intScheme->getEx ()->getSize () - yeeLayout->getExEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getEx ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::EX,
                                                       intScheme->getEps (), GridType::EPS,
                                                       intScheme->getOmegaPE (), GridType::OMEGAPE,
                                                       intScheme->getGammaE (), GridType::GAMMAE,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Eps0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getCB0Ex ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getCB1Ex ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getCB2Ex ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getCA1Ex ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getCA2Ex ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEyStartDiff () && pos < intScheme->getEy ()->getSize () - yeeLayout->getEyEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getEy ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::EY,
                                                       intScheme->getEps (), GridType::EPS,
                                                       intScheme->getOmegaPE (), GridType::OMEGAPE,
                                                       intScheme->getGammaE (), GridType::GAMMAE,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Eps0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getCB0Ey ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getCB1Ey ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getCB2Ey ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getCA1Ey ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getCA2Ey ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getEz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getEz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getEzStartDiff () && pos < intScheme->getEz ()->getSize () - yeeLayout->getEzEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getEz ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::EZ,
                                                       intScheme->getEps (), GridType::EPS,
                                                       intScheme->getOmegaPE (), GridType::OMEGAPE,
                                                       intScheme->getGammaE (), GridType::GAMMAE,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Eps0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getCB0Ez ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getCB1Ez ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getCB2Ez ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getCA1Ez ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getCA2Ez ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHx ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHx ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHxStartDiff () && pos < intScheme->getHx ()->getSize () - yeeLayout->getHxEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getHx ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::HX,
                                                       intScheme->getMu (), GridType::MU,
                                                       intScheme->getOmegaPM (), GridType::OMEGAPM,
                                                       intScheme->getGammaM (), GridType::GAMMAM,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Mu0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getDB0Hx ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getDB1Hx ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getDB2Hx ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getDA1Hx ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getDA2Hx ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHy ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHy ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHyStartDiff () && pos < intScheme->getHy ()->getSize () - yeeLayout->getHyEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getHy ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::HY,
                                                       intScheme->getMu (), GridType::MU,
                                                       intScheme->getOmegaPM (), GridType::OMEGAPM,
                                                       intScheme->getGammaM (), GridType::GAMMAM,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Mu0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getDB0Hy ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getDB1Hy ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getDB2Hy ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getDA1Hy ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getDA2Hy ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      typename VectorFieldValues<TC>::Iterator iter = intScheme->getHz ()->begin ();
      typename VectorFieldValues<TC>::Iterator iter_end = intScheme->getHz ()->end ();
      for (; iter != iter_end; ++iter)
      {
        TC pos = iter.getPos ();

        if (!(pos >= yeeLayout->getHzStartDiff () && pos < intScheme->getHz ()->getSize () - yeeLayout->getHzEndDiff ()))
        {
          continue;
        }

        TC posAbs = intScheme->getHz ()->getTotalPosition (pos);

        FPValue material1;
        FPValue material2;
        FPValue material = intScheme->getMetaMaterial (posAbs, GridType::HZ,
                                                       intScheme->getMu (), GridType::MU,
                                                       intScheme->getOmegaPM (), GridType::OMEGAPM,
                                                       intScheme->getGammaM (), GridType::GAMMAM,
                                                       material1, material2);

        FPValue materialModifier = PhysicsConst::Mu0;
        FPValue A = 4*materialModifier*material + 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1);
        FPValue b0 = (4 + 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue b1 = -8 / A;
        FPValue b2 = (4 - 2*intScheme->getGridTimeStep ()*material2) / A;
        FPValue a1 = (2*materialModifier*SQR(intScheme->getGridTimeStep ()*material1) - 8*materialModifier*material) / A;
        FPValue a2 = (4*materialModifier*material - 2*intScheme->getGridTimeStep ()*materialModifier*material*material2 + materialModifier*SQR(intScheme->getGridTimeStep ()*material1)) / A;

        intScheme->getDB0Hz ()->setFieldValue (FIELDVALUE (b0, FPValue (0)), pos, 0);
        intScheme->getDB1Hz ()->setFieldValue (FIELDVALUE (b1, FPValue (0)), pos, 0);
        intScheme->getDB2Hz ()->setFieldValue (FIELDVALUE (b2, FPValue (0)), pos, 0);
        intScheme->getDA1Hz ()->setFieldValue (FIELDVALUE (a1, FPValue (0)), pos, 0);
        intScheme->getDA2Hz ()->setFieldValue (FIELDVALUE (a2, FPValue (0)), pos, 0);
      }
    }
  }
}

/**
 * Compute value of time-averaged Poynting vector of the scattered field (mulptiplied on 4*Pi*r^2)
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *curEx, Grid<TC> *curEy, Grid<TC> *curEz,
                       Grid<TC> *curHx, Grid<TC> *curHy, Grid<TC> *curHz, TC leftNTFF, TC rightNTFF)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue k = 2 * PhysicsConst::Pi / intScheme->getSourceWaveLength (); // TODO: check numerical here

  NPair N = ntffN (angleTeta, anglePhi, curEx, curEy, curEz, curHx, curHy, curHz, leftNTFF, rightNTFF);
  NPair L = ntffL (angleTeta, anglePhi, curEx, curEy, curEz, curHx, curHy, curHz, leftNTFF, rightNTFF);

  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();

    FieldValue tmpArray[4];
    FieldValue tmpArrayRes[4];
    const int count = 4;

    tmpArray[0] = N.nTeta;
    tmpArray[1] = N.nPhi;
    tmpArray[2] = L.nTeta;
    tmpArray[3] = L.nPhi;

    // gather all sum_teta and sum_phi on 0 node
    MPI_Reduce (tmpArray, tmpArrayRes, count, MPI_FPVALUE, MPI_SUM, 0, ParallelGrid::getParallelCore ()->getCommunicator ());

    if (processId == 0)
    {
      N.nTeta = FieldValue (tmpArrayRes[0]);
      N.nPhi = FieldValue (tmpArrayRes[1]);

      L.nTeta = FieldValue (tmpArrayRes[2]);
      L.nPhi = FieldValue (tmpArrayRes[3]);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  if (processId == 0)
  {
    FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);

    FieldValue first = L.nPhi + N.nTeta * n0;
    FieldValue second = L.nTeta - N.nPhi * n0;

    FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
    FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());

    return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
  }
  else
  {
    return FPValue (0);
  }
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
  return FPValue (0);
#endif
}

/**
 * Compute value of time-averaged Poynting vector of the incident field
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
{
  // TODO: consider amplitude here, i.e. compute |E|^2 of incident field, instead of considering it 1.0
  return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0) / FPValue (2);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::makeGridScattered (Grid<TC> *grid, GridType gridType)
{
  typename VectorFieldValues<TC>::Iterator iter = grid->begin ();
  typename VectorFieldValues<TC>::Iterator iter_end = grid->end ();
  for (; iter != iter_end; ++iter)
  {
    TC pos = iter.getPos ();
    FieldValue *val = grid->getFieldValue (pos, 1);
    TC posAbs = grid->getTotalPosition (pos);

    TCFP realCoord;
    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doSkipMakeScattered (realCoord))
    {
      continue;
    }

    FieldValue iVal;
    if (gridType == GridType::EX
        || gridType == GridType::EY
        || gridType == GridType::EZ)
    {
      iVal = intScheme->approximateIncidentWaveE (realCoord);
    }
    else if (gridType == GridType::HX
             || gridType == GridType::HY
             || gridType == GridType::HZ)
    {
      iVal = intScheme->approximateIncidentWaveH (realCoord);
    }
    else
    {
      UNREACHABLE;
    }

    FieldValue incVal;
    switch (gridType)
    {
      case GridType::EX:
      {
        incVal = yeeLayout->getExFromIncidentE (iVal);
        break;
      }
      case GridType::EY:
      {
        incVal = yeeLayout->getEyFromIncidentE (iVal);
        break;
      }
      case GridType::EZ:
      {
        incVal = yeeLayout->getEzFromIncidentE (iVal);
        break;
      }
      case GridType::HX:
      {
        incVal = yeeLayout->getHxFromIncidentH (iVal);
        break;
      }
      case GridType::HY:
      {
        incVal = yeeLayout->getHyFromIncidentH (iVal);
        break;
      }
      case GridType::HZ:
      {
        incVal = yeeLayout->getHzFromIncidentH (iVal);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    *val = *val - incVal;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::gatherFieldsTotal (bool scattered)
{
  if (useParallel)
  {
    initFullFieldGrids ();
  }
  else
  {
    if (totalInitialized)
    {
      if (intScheme->getDoNeedEx ())
      {
        totalEx->copy (intScheme->getEx ());
      }
      if (intScheme->getDoNeedEy ())
      {
        totalEy->copy (intScheme->getEy ());
      }
      if (intScheme->getDoNeedEz ())
      {
        totalEz->copy (intScheme->getEz ());
      }

      if (intScheme->getDoNeedHx ())
      {
        totalHx->copy (intScheme->getHx ());
      }
      if (intScheme->getDoNeedHy ())
      {
        totalHy->copy (intScheme->getHy ());
      }
      if (intScheme->getDoNeedHz ())
      {
        totalHz->copy (intScheme->getHz ());
      }
    }
    else
    {
      if (scattered)
      {
        if (intScheme->getDoNeedEx ())
        {
          totalEx = new Grid<TC> (yeeLayout->getExSize (), intScheme->getEx ()->getCountStoredSteps (), "Ex");
          totalEx->copy (intScheme->getEx ());
        }
        if (intScheme->getDoNeedEy ())
        {
          totalEy = new Grid<TC> (yeeLayout->getEySize (), intScheme->getEy ()->getCountStoredSteps (), "Ey");
          totalEy->copy (intScheme->getEy ());
        }
        if (intScheme->getDoNeedEz ())
        {
          totalEz = new Grid<TC> (yeeLayout->getEzSize (), intScheme->getEz ()->getCountStoredSteps (), "Ez");
          totalEz->copy (intScheme->getEz ());
        }

        if (intScheme->getDoNeedHx ())
        {
          totalHx = new Grid<TC> (yeeLayout->getHxSize (), intScheme->getHx ()->getCountStoredSteps (), "Hx");
          totalHx->copy (intScheme->getHx ());
        }
        if (intScheme->getDoNeedHy ())
        {
          totalHy = new Grid<TC> (yeeLayout->getHySize (), intScheme->getHy ()->getCountStoredSteps (), "Hy");
          totalHy->copy (intScheme->getHy ());
        }
        if (intScheme->getDoNeedHz ())
        {
          totalHz = new Grid<TC> (yeeLayout->getHzSize (), intScheme->getHz ()->getCountStoredSteps (), "Hz");
          totalHz->copy (intScheme->getHz ());
        }

        totalInitialized = true;
      }
      else
      {
        if (intScheme->getDoNeedEx ())
        {
          totalEx = intScheme->getEx ();
        }
        if (intScheme->getDoNeedEy ())
        {
          totalEy = intScheme->getEy ();
        }
        if (intScheme->getDoNeedEz ())
        {
          totalEz = intScheme->getEz ();
        }

        if (intScheme->getDoNeedHx ())
        {
          totalHx = intScheme->getHx ();
        }
        if (intScheme->getDoNeedHy ())
        {
          totalHy = intScheme->getHy ();
        }
        if (intScheme->getDoNeedHz ())
        {
          totalHz = intScheme->getHz ();
        }
      }
    }
  }

  if (scattered)
  {
    if (intScheme->getDoNeedEx ())
    {
      makeGridScattered (totalEx, GridType::EX);
    }
    if (intScheme->getDoNeedEy ())
    {
      makeGridScattered (totalEy, GridType::EY);
    }
    if (intScheme->getDoNeedEz ())
    {
      makeGridScattered (totalEz, GridType::EZ);
    }

    if (intScheme->getDoNeedHx ())
    {
      makeGridScattered (totalHx, GridType::HX);
    }
    if (intScheme->getDoNeedHy ())
    {
      makeGridScattered (totalHy, GridType::HY);
    }
    if (intScheme->getDoNeedHz ())
    {
      makeGridScattered (totalHz, GridType::HZ);
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveGrids (time_step t)
{
  int processId = 0;
  if (SOLVER_SETTINGS.getDoSaveResPerProcess ()
      && useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    UNREACHABLE;
#endif
  }

  TC startEx;
  TC endEx;
  TC startEy;
  TC endEy;
  TC startEz;
  TC endEz;
  TC startHx;
  TC endHx;
  TC startHy;
  TC endHy;
  TC startHz;
  TC endHz;

  TC zero = TC_COORD (0, 0, 0, ct1, ct2, ct3);

  if (SOLVER_SETTINGS.getDoUseManualStartEndDumpCoord ())
  {
    TC start = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveStartCoordX (),
                                       SOLVER_SETTINGS.getSaveStartCoordY (),
                                       SOLVER_SETTINGS.getSaveStartCoordZ (),
                                       ct1, ct2, ct3);
    TC end = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveEndCoordX (),
                                     SOLVER_SETTINGS.getSaveEndCoordY (),
                                     SOLVER_SETTINGS.getSaveEndCoordZ (),
                                     ct1, ct2, ct3);

    startEx = startEy = startEz = startHx = startHy = startHz = start;
    endEx = endEy = endEz = endHx = endHy = endHz = end;
  }
  else
  {
    startEx = intScheme->getDoNeedEx () ? getStartCoord (GridType::EX, intScheme->getEx ()->getTotalSize ()) : zero;
    endEx = intScheme->getDoNeedEx () ? getEndCoord (GridType::EX, intScheme->getEx ()->getTotalSize ()) : zero;

    startEy = intScheme->getDoNeedEy () ? getStartCoord (GridType::EY, intScheme->getEy ()->getTotalSize ()) : zero;
    endEy = intScheme->getDoNeedEy () ? getEndCoord (GridType::EY, intScheme->getEy ()->getTotalSize ()) : zero;

    startEz = intScheme->getDoNeedEz () ? getStartCoord (GridType::EZ, intScheme->getEz ()->getTotalSize ()) : zero;
    endEz = intScheme->getDoNeedEz () ? getEndCoord (GridType::EZ, intScheme->getEz ()->getTotalSize ()) : zero;

    startHx = intScheme->getDoNeedHx () ? getStartCoord (GridType::HX, intScheme->getHx ()->getTotalSize ()) : zero;
    endHx = intScheme->getDoNeedHx () ? getEndCoord (GridType::HX, intScheme->getHx ()->getTotalSize ()) : zero;

    startHy = intScheme->getDoNeedHy () ? getStartCoord (GridType::HY, intScheme->getHy ()->getTotalSize ()) : zero;
    endHy = intScheme->getDoNeedHy () ? getEndCoord (GridType::HY, intScheme->getHy ()->getTotalSize ()) : zero;

    startHz = intScheme->getDoNeedHz () ? getStartCoord (GridType::HZ, intScheme->getHz ()->getTotalSize ()) : zero;
    endHz = intScheme->getDoNeedHz () ? getEndCoord (GridType::HZ, intScheme->getHz ()->getTotalSize ()) : zero;
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    int currentLayer = 1;

    if (intScheme->getDoNeedEx ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEx (), zero, intScheme->getEx ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalEx, startEx, endEx, t, currentLayer, processId);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEy (), zero, intScheme->getEy ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalEy, startEy, endEy, t, currentLayer, processId);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEz (), zero, intScheme->getEz ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalEz, startEz, endEz, t, currentLayer, processId);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHx (), zero, intScheme->getHx ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalHx, startHx, endHx, t, currentLayer, processId);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHy (), zero, intScheme->getHy ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalHy, startHy, endHy, t, currentLayer, processId);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHz (), zero, intScheme->getHz ()->getSize (), t, currentLayer, processId);
      }
      else if (processId == 0)
      {
        dumper[type]->dumpGrid (totalHz, startHz, endHz, t, currentLayer, processId);
      }
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFEInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->dumpGrid (intScheme->getEInc (), GRID_COORDINATE_1D (0, CoordinateType::X), intScheme->getEInc ()->getSize (), t, currentLayer, processId);
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFHInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->dumpGrid (intScheme->getHInc (), GRID_COORDINATE_1D (0, CoordinateType::X), intScheme->getHInc ()->getSize (), t, currentLayer, processId);
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveNTFF (bool isReverse, time_step t)
{
  DPRINTF (LOG_LEVEL_STAGES, "Saving NTFF.\n");

  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  for (grid_coord step_ntff = 0; step_ntff < SOLVER_SETTINGS.getNTFFDiff (); ++step_ntff)
  {
    TC stepNTFF = TC::initAxesCoordinate (step_ntff, step_ntff, step_ntff, ct1, ct2, ct3);
    TC leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                          ct1, ct2, ct3);
    TC rightNTFF = yeeLayout->getSize () - leftNTFF + TC_COORD (1, 1, 1, ct1, ct2, ct3);

    leftNTFF = leftNTFF + stepNTFF;
    rightNTFF = rightNTFF - stepNTFF;

    std::ofstream outfile;
    std::ostream *outs;
    const char *strName;
    FPValue start;
    FPValue end;
    FPValue step;

    if (isReverse)
    {
      strName = "Reverse diagram";
      start = yeeLayout->getIncidentWaveAngle2 ();
      end = yeeLayout->getIncidentWaveAngle2 ();
      step = 1.0;
    }
    else
    {
      strName = "Forward diagram";
      start = 0.0;
      end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
      step = PhysicsConst::Pi * SOLVER_SETTINGS.getAngleStepNTFF () / 180;
    }

    if (processId == 0)
    {
      if (SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
      {
        outs = &std::cout;
        DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saving NTFF[%d] to stdout.\n", step_ntff);
      }
      else
      {
        std::string ntffFileName = SOLVER_SETTINGS.getFileNameNTFF ()
                                   + std::string ("_[timestep=")
                                   + int64_to_string (t)
                                   + std::string ("]_[step=")
                                   + int64_to_string (step_ntff)
                                   + std::string ("].txt");
        outfile.open (ntffFileName.c_str ());
        outs = &outfile;
        DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Saving NTFF[%d] to %s.\n", step_ntff, ntffFileName.c_str ());
      }
      (*outs) << "==== NTFF (step=" << step_ntff << ") ====" << std::endl;
      (*outs) << strName << std::endl << std::endl;
    }

    for (FPValue angle = start; angle <= end; angle += step)
    {
      FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
                                   angle,
                                   intScheme->getEx (),
                                   intScheme->getEy (),
                                   intScheme->getEz (),
                                   intScheme->getHx (),
                                   intScheme->getHy (),
                                   intScheme->getHz (),
                                   leftNTFF,
                                   rightNTFF);
      val /= Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);

      if (processId == 0)
      {
        (*outs) << "timestep = "
                << t
                << ", incident wave angle=("
                << yeeLayout->getIncidentWaveAngle1 () << ","
                << yeeLayout->getIncidentWaveAngle2 () << ","
                << yeeLayout->getIncidentWaveAngle3 () << ","
                << "), angle NTFF = "
                << angle
                << ", NTFF value = "
                << val
                << std::endl;
      }
    }

    if (processId == 0)
    {
      if (!SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
      {
        outfile.close ();
      }
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getStartCoord (GridType gridType, TC size)
{
  TC start (0, 0, 0
#ifdef DEBUG_INFO
            , ct1, ct2, ct3
#endif
            );

  if (SOLVER_SETTINGS.getDoSaveWithoutPML ()
      && SOLVER_SETTINGS.getDoUsePML ())
  {
    TCFP leftBorder = convertCoord (yeeLayout->getLeftBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    start = convertCoord (expandTo3D (leftBorder - min, ct1, ct2, ct3)) + GridCoordinate3D (1, 1, 1
#ifdef DEBUG_INFO
                                                                                            , ct1, ct2, ct3
#endif
                                                                                            );
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getStartCoordRes (orthogonalAxis, start, size);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getEndCoord (GridType gridType, TC size)
{
  TC end = size;
  if (SOLVER_SETTINGS.getDoSaveWithoutPML ()
      && SOLVER_SETTINGS.getDoUsePML ())
  {
    TCFP rightBorder = convertCoord (yeeLayout->getRightBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    end = convertCoord (expandTo3D (rightBorder - min, ct1, ct2, ct3));
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getEndCoordRes (orthogonalAxis, end, size);
}

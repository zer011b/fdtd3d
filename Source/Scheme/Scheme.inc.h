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
  /*
   * Copy InternalScheme to GPU
   */
  gpuIntScheme->copyFromCPU (blockIdx * blockSize, blockSize);
  gpuIntSchemeOnGPU->copyToGPU (gpuIntScheme);
  cudaCheckErrorCmd (cudaMemcpy (d_gpuIntSchemeOnGPU, gpuIntSchemeOnGPU, sizeof(InternalSchemeGPU<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));
#endif /* CUDA_ENABLED */

  for (time_step t = tStart; t < tStart + N; ++t)
  {
    if (processId == 0)
    {
      DPRINTF (LOG_LEVEL_STAGES, "Calculating time step %u...\n", t);
    }

#ifdef CUDA_ENABLED
    TC ExStart = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = gpuIntScheme->getDoNeedEx () ? gpuIntScheme->getEx ()->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = gpuIntScheme->getDoNeedEy () ? gpuIntScheme->getEy ()->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = gpuIntScheme->getDoNeedEz () ? gpuIntScheme->getEz ()->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = gpuIntScheme->getDoNeedHx () ? gpuIntScheme->getHx ()->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = gpuIntScheme->getDoNeedHy () ? gpuIntScheme->getHy ()->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = gpuIntScheme->getDoNeedHz () ? gpuIntScheme->getHz ()->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
#else /* CUDA_ENABLED */
    TC ExStart = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationStart (yeeLayout->getExStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC ExEnd = intScheme->getDoNeedEx () ? intScheme->getEx ()->getComputationEnd (yeeLayout->getExEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EyStart = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationStart (yeeLayout->getEyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EyEnd = intScheme->getDoNeedEy () ? intScheme->getEy ()->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC EzStart = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationStart (yeeLayout->getEzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC EzEnd = intScheme->getDoNeedEz () ? intScheme->getEz ()->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HxStart = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationStart (yeeLayout->getHxStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HxEnd = intScheme->getDoNeedHx () ? intScheme->getHx ()->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HyStart = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationStart (yeeLayout->getHyStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HyEnd = intScheme->getDoNeedHy () ? intScheme->getHy ()->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);

    TC HzStart = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationStart (yeeLayout->getHzStartDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
    TC HzEnd = intScheme->getDoNeedHz () ? intScheme->getHz ()->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC_COORD (0, 0, 0, ct1, ct2, ct3);
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

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#ifdef CUDA_ENABLED
      ASSERT_MESSAGE ("Balancing is WIP for Cuda builds");
#endif

#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
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
#ifdef CUDA_ENABLED
      ASSERT_MESSAGE ("Balancing is WIP for Cuda builds");
#endif

#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedEx ())
    {
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

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#ifdef CUDA_ENABLED
      ASSERT_MESSAGE ("Balancing is WIP for Cuda builds");
#endif

#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
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
#ifdef CUDA_ENABLED
      ASSERT_MESSAGE ("Balancing is WIP for Cuda builds");
#endif

#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (intScheme->getDoNeedHx ())
    {
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
  //       if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  //       {
  //         if (internalScheme.doNeedEx)
  //         {
  //           ((ParallelGrid *) internalScheme.ExAmplitude)->Resize (parallelYeeLayout->getExSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedEy)
  //         {
  //           ((ParallelGrid *) internalScheme.EyAmplitude)->Resize (parallelYeeLayout->getEySizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedEz)
  //         {
  //           ((ParallelGrid *) internalScheme.EzAmplitude)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
  //         }
  //
  //         if (internalScheme.doNeedHx)
  //         {
  //           ((ParallelGrid *) internalScheme.HxAmplitude)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedHy)
  //         {
  //           ((ParallelGrid *) internalScheme.HyAmplitude)->Resize (parallelYeeLayout->getHySizeForCurNode ());
  //         }
  //         if (internalScheme.doNeedHz)
  //         {
  //           ((ParallelGrid *) internalScheme.HzAmplitude)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
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
    gpuIntSchemeOnGPU->performPointSourceCalcKernelLaunch<grid_type> (d_gpuIntSchemeOnGPU, t);
#else /* CUDA_ENABLED */
    intScheme->template performPointSourceCalc<grid_type> (t);
#endif /* !CUDA_ENABLED */
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

  gpuIntSchemeOnGPU->template calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&d_grid, &_gridType,
    &d_materialGrid, &_materialGridType, &d_materialGrid1, &_materialGridType1, &d_materialGrid2, &_materialGridType2,
    &d_materialGrid3, &_materialGridType3, &d_materialGrid4, &_materialGridType4, &d_materialGrid5, &_materialGridType5,
    &d_oppositeGrid1, &d_oppositeGrid2, &d_gridPML1, &_gridPMLType1, &d_gridPML2, &_gridPMLType2,
    &_rightSideFunc, &_borderFunc, &_exactFunc, &_materialModifier, &d_Ca, &d_Cb,
    &d_CB0, &d_CB1, &d_CB2, &d_CA1, &d_CA2, &d_CaPML, &d_CbPML, &d_CcPML);

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
    gpuIntSchemeOnGPU->template calculateFieldStepIterationKernelLaunch <grid_type>
      (d_gpuIntSchemeOnGPU, start3D, end3D,
        t, diff11, diff12, diff21, diff22,
        d_grid,
        d_oppositeGrid1,
        d_oppositeGrid2,
        rightSideFunc,
        d_Ca,
        d_Cb,
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
            intScheme->template calculateFieldStepIteration<grid_type, true> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               grid, coordFP,
                                                               oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                               usePML,
                                                               gridType, materialGrid, materialGridType,
                                                               materialModifier);
          }
          else
          {
            intScheme->template calculateFieldStepIteration<grid_type, false> (t, pos, posAbs, diff11, diff12, diff21, diff22,
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
           t, d_grid, d_gridPML1,
           d_CB0, d_CB1, d_CB2, d_CA1, d_CA2,
           gridType,
           d_materialGrid1, materialGridType1,
           d_materialGrid2, materialGridType2,
           d_materialGrid3, materialGridType3,
           materialModifier,
           SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ());

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

#endif /* !CUDA_ENABLED */
      }

#ifdef CUDA_ENABLED

      gpuIntSchemeOnGPU->template calculateFieldStepIterationPMLKernelLaunch <useMetamaterials> (d_gpuIntSchemeOnGPU, start3D, end3D,
        t, d_grid, d_gridPML1, d_gridPML2, d_CaPML, d_CbPML, d_CcPML, gridPMLType1,
        d_materialGrid1, materialGridType1, d_materialGrid4, materialGridType4, d_materialGrid5, materialGridType5,
        materialModifier,
        SOLVER_SETTINGS.getDoUseCaCbPMLGrids ());

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

#endif /* !CUDA_ENABLED */
    }
  }

  if (borderFunc != NULLPTR)
  {
    GridCoordinate3D startBorder;
    GridCoordinate3D endBorder;

    expandTo3DStartEnd (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3),
                        grid->getSize (),
                        startBorder,
                        endBorder,
                        ct1, ct2, ct3);

#ifdef CUDA_ENABLED

    gpuIntSchemeOnGPU->template calculateFieldStepIterationBorderKernelLaunch<grid_type>
      (d_gpuIntSchemeOnGPU, startBorder, endBorder, t, d_grid, borderFunc);

#else

    for (grid_coord i = startBorder.get1 (); i < endBorder.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startBorder.get2 (); j < endBorder.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startBorder.get3 (); k < endBorder.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          intScheme->template calculateFieldStepIterationBorder<grid_type> (t, pos, grid, borderFunc);
        }
      }
    }
#endif
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

    Grid<TC> *normGrid = grid;
    if (usePML)
    {
      normGrid = gridPML2;
    }

#ifdef CUDA_ENABLED

    CudaGrid<TC> *d_normGrid = d_grid;
    if (usePML)
    {
      d_normGrid = d_gridPML2;
    }

    gpuIntSchemeOnGPU->template calculateFieldStepIterationExactKernelLaunch<grid_type>
      (d_gpuIntSchemeOnGPU, startNorm, endNorm, t, d_normGrid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);

#else /* CUDA_ENABLED */

    for (grid_coord i = startNorm.get1 (); i < endNorm.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startNorm.get2 (); j < endNorm.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startNorm.get3 (); k < endNorm.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          intScheme->template calculateFieldStepIterationExact<grid_type> (t, pos, normGrid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
        }
      }
    }
#endif /* CUDA_ENABLED */

#ifdef COMPLEX_FIELD_VALUES
    normRe = sqrt (normRe / normGrid->getSize ().calculateTotalCoord ());
    normIm = sqrt (normIm / normGrid->getSize ().calculateTotalCoord ());
    normMod = sqrt (normMod / normGrid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
      normGrid->getName (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
    normRe = sqrt (normRe / normGrid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
      normGrid->getName (), t, normRe, normRe * 100.0 / maxRe);
#endif
  }
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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                           bool parallelLayout,
                                           const TC& totSize,
                                           time_step tStep)
  : useParallel (false)
  , intScheme (new InternalScheme<Type, TCoord, layout_type> ())
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

  ASSERT (!SOLVER_SETTINGS.getDoUseAmplitudeMode ()
          || SOLVER_SETTINGS.getDoUseAmplitudeMode () && SOLVER_SETTINGS.getNumAmplitudeSteps () != 0);

#ifdef COMPLEX_FIELD_VALUES
  ASSERT (!SOLVER_SETTINGS.getDoUseAmplitudeMode ());
#endif /* COMPLEX_FIELD_VALUES */

  if (SOLVER_SETTINGS.getDoUseParallelGrid ())
  {
#ifndef PARALLEL_GRID
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.")
#endif

    ALWAYS_ASSERT (parallelLayout);
#ifdef PARALLEL_GRID
    ALWAYS_ASSERT ((TCoord<grid_coord, false>::dimension == ParallelGridCoordinateTemplate<grid_coord, false>::dimension));
#endif

    useParallel = true;
  }

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
    /*
     * In parallel mode total grids will be allocated if required
     */
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

//
// template <SchemeType_t Type, template <typename, bool> class TCoord, typename Layout>
// void
// Scheme<Type, TCoord, Layout>::performAmplitudeSteps (time_step startStep)
// {
// #ifdef COMPLEX_FIELD_VALUES
//   UNREACHABLE;
// #else /* COMPLEX_FIELD_VALUES */
//
//   ASSERT_MESSAGE ("Temporary unsupported");
//
//   int processId = 0;
//
//   if (SOLVER_SETTINGS.getDoUseParallelGrid ())
//   {
// #ifdef PARALLEL_GRID
//     processId = ParallelGrid::getParallelCore ()->getProcessId ();
// #else /* PARALLEL_GRID */
//     ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
// #endif /* !PARALLEL_GRID */
//   }
//
//   int is_stable_state = 0;
//
//   GridCoordinate3D EzSize = internalScheme.Ez->getSize ();
//
//   time_step t = startStep;
//
//   while (is_stable_state == 0 && t < SOLVER_SETTINGS.getNumAmplitudeSteps ())
//   {
//     FPValue maxAccuracy = -1;
//
//     //is_stable_state = 1;
//
//     GridCoordinate3D ExStart = internalScheme.Ex->getComputationStart (yeeLayout->getExStartDiff ());
//     GridCoordinate3D ExEnd = internalScheme.Ex->getComputationEnd (yeeLayout->getExEndDiff ());
//
//     GridCoordinate3D EyStart = internalScheme.Ey->getComputationStart (yeeLayout->getEyStartDiff ());
//     GridCoordinate3D EyEnd = internalScheme.Ey->getComputationEnd (yeeLayout->getEyEndDiff ());
//
//     GridCoordinate3D EzStart = internalScheme.Ez->getComputationStart (yeeLayout->getEzStartDiff ());
//     GridCoordinate3D EzEnd = internalScheme.Ez->getComputationEnd (yeeLayout->getEzEndDiff ());
//
//     GridCoordinate3D HxStart = internalScheme.Hx->getComputationStart (yeeLayout->getHxStartDiff ());
//     GridCoordinate3D HxEnd = internalScheme.Hx->getComputationEnd (yeeLayout->getHxEndDiff ());
//
//     GridCoordinate3D HyStart = internalScheme.Hy->getComputationStart (yeeLayout->getHyStartDiff ());
//     GridCoordinate3D HyEnd = internalScheme.Hy->getComputationEnd (yeeLayout->getHyEndDiff ());
//
//     GridCoordinate3D HzStart = internalScheme.Hz->getComputationStart (yeeLayout->getHzStartDiff ());
//     GridCoordinate3D HzEnd = internalScheme.Hz->getComputationEnd (yeeLayout->getHzEndDiff ());
//
//     if (SOLVER_SETTINGS.getDoUseTFSF ())
//     {
//       performPlaneWaveESteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
//
//     for (int i = ExStart.get1 (); i < ExEnd.get1 (); ++i)
//     {
//       for (int j = ExStart.get2 (); j < ExEnd.get2 (); ++j)
//       {
//         for (int k = ExStart.get3 (); k < ExEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isExInPML (internalScheme.Ex->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ex->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.ExAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (internalScheme.Ex->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EyStart.get1 (); i < EyEnd.get1 (); ++i)
//     {
//       for (int j = EyStart.get2 (); j < EyEnd.get2 (); ++j)
//       {
//         for (int k = EyStart.get3 (); k < EyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEyInPML (internalScheme.Ey->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ey->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.EyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (internalScheme.Ey->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EzStart.get1 (); i < EzEnd.get1 (); ++i)
//     {
//       for (int j = EzStart.get2 (); j < EzEnd.get2 (); ++j)
//       {
//         for (int k = EzStart.get3 (); k < EzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEzInPML (internalScheme.Ez->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ez->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.EzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (internalScheme.Ez->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     internalScheme.Ex->nextTimeStep ();
//     internalScheme.Ey->nextTimeStep ();
//     internalScheme.Ez->nextTimeStep ();
//
//     if (SOLVER_SETTINGS.getDoUsePML ())
//     {
//       internalScheme.Dx->nextTimeStep ();
//       internalScheme.Dy->nextTimeStep ();
//       internalScheme.Dz->nextTimeStep ();
//     }
//
//     if (SOLVER_SETTINGS.getDoUseTFSF ())
//     {
//       performPlaneWaveHSteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
//
//     for (int i = HxStart.get1 (); i < HxEnd.get1 (); ++i)
//     {
//       for (int j = HxStart.get2 (); j < HxEnd.get2 (); ++j)
//       {
//         for (int k = HxStart.get3 (); k < HxEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHxInPML (internalScheme.Hx->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hx->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HxAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (internalScheme.Hx->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HyStart.get1 (); i < HyEnd.get1 (); ++i)
//     {
//       for (int j = HyStart.get2 (); j < HyEnd.get2 (); ++j)
//       {
//         for (int k = HyStart.get3 (); k < HyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHyInPML (internalScheme.Hy->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hy->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (internalScheme.Hy->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HzStart.get1 (); i < HzEnd.get1 (); ++i)
//     {
//       for (int j = HzStart.get2 (); j < HzEnd.get2 (); ++j)
//       {
//         for (int k = HzStart.get3 (); k < HzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHzInPML (internalScheme.Hz->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hz->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (internalScheme.Hz->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     internalScheme.Hx->nextTimeStep ();
//     internalScheme.Hy->nextTimeStep ();
//     internalScheme.Hz->nextTimeStep ();
//
//     if (SOLVER_SETTINGS.getDoUsePML ())
//     {
//       internalScheme.Bx->nextTimeStep ();
//       internalScheme.By->nextTimeStep ();
//       internalScheme.Bz->nextTimeStep ();
//     }
//
//     ++t;
//
//     if (maxAccuracy < 0)
//     {
//       is_stable_state = 0;
//     }
//
//     DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " FP_MOD ". \n", t, maxAccuracy);
//   }
//
//   if (is_stable_state == 0)
//   {
//     ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.");
//   }
//
// #endif /* !COMPLEX_FIELD_VALUES */
// }

// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// int
// Scheme<Type, TCoord, layout_type>::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
// {
// #ifdef COMPLEX_FIELD_VALUES
//   UNREACHABLE;
// #else /* COMPLEX_FIELD_VALUES */
//
//   int is_stable_state = 1;
//
//   FPValue valAmp = amplitudeValue->getCurValue ();
//
//   val = val >= 0 ? val : -val;
//
//   if (val >= valAmp)
//   {
//     FPValue accuracy = val - valAmp;
//     if (valAmp != 0)
//     {
//       accuracy /= valAmp;
//     }
//     else if (val != 0)
//     {
//       accuracy /= val;
//     }
//
//     if (accuracy > PhysicsConst::accuracy)
//     {
//       is_stable_state = 0;
//
//       amplitudeValue->setCurValue (val);
//     }
//
//     if (accuracy > *maxAccuracy)
//     {
//       *maxAccuracy = accuracy;
//     }
//   }
//
//   return is_stable_state;
// #endif /* !COMPLEX_FIELD_VALUES */
// }

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
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&tmp, CallBack::polinom1_ez, sizeof(SourceCallBack));
    intScheme->setCallbackEzBorder (tmp);
    cudaMemcpyFromSymbol (&tmp, CallBack::polinom1_hy, sizeof(SourceCallBack));
    intScheme->setCallbackHyBorder (tmp);
#else
    intScheme->setCallbackEzBorder (CallBack::polinom1_ez);
    intScheme->setCallbackHyBorder (CallBack::polinom1_hy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2BorderCondition ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExBorder, CallBack::polinom2_ex, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&EyBorder, CallBack::polinom2_ey, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&EzBorder, CallBack::polinom2_ez, sizeof(SourceCallBack));

    cudaMemcpyFromSymbol (&HxBorder, CallBack::polinom2_hx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyBorder, CallBack::polinom2_hy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HzBorder, CallBack::polinom2_hz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExBorder (CallBack::polinom2_ex);
    intScheme->setCallbackEyBorder (CallBack::polinom2_ey);
    intScheme->setCallbackEzBorder (CallBack::polinom2_ez);

    intScheme->setCallbackHxBorder (CallBack::polinom2_hx);
    intScheme->setCallbackHyBorder (CallBack::polinom2_hy);
    intScheme->setCallbackHzBorder (CallBack::polinom2_hz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3BorderCondition ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzBorder, CallBack::polinom3_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyBorder, CallBack::polinom3_hy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzBorder (CallBack::polinom3_ez);
    intScheme->setCallbackHyBorder (CallBack::polinom3_hy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoUseSin1BorderCondition ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzBorder, CallBack::sin1_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyBorder, CallBack::sin1_hy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzBorder (CallBack::sin1_ez);
    intScheme->setCallbackHyBorder (CallBack::sin1_hy);
#endif
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
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&Jz, CallBack::polinom1_jz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&My, CallBack::polinom1_my, sizeof(SourceCallBack));
#else
    intScheme->setCallbackJz (CallBack::polinom1_jz);
    intScheme->setCallbackMy (CallBack::polinom1_my);
#endif /* !CUDA_ENABLED */
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2RightSide ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&Jx, CallBack::polinom2_jx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&Jy, CallBack::polinom2_jy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&Jz, CallBack::polinom2_jz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&Mx, CallBack::polinom2_mx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&My, CallBack::polinom2_my, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&Mz, CallBack::polinom2_mz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackJx (CallBack::polinom2_jx);
    intScheme->setCallbackJy (CallBack::polinom2_jy);
    intScheme->setCallbackJz (CallBack::polinom2_jz);

    intScheme->setCallbackMx (CallBack::polinom2_mx);
    intScheme->setCallbackMy (CallBack::polinom2_my);
    intScheme->setCallbackMz (CallBack::polinom2_mz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3RightSide ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&Jz, CallBack::polinom3_jz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&My, CallBack::polinom3_my, sizeof(SourceCallBack));
#else
    intScheme->setCallbackJz (CallBack::polinom3_jz);
    intScheme->setCallbackMy (CallBack::polinom3_my);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculatePolinom1DiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::polinom1_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::polinom1_hy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::polinom1_ez);
    intScheme->setCallbackHyExact (CallBack::polinom1_hy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom2DiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::polinom2_ex, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&EyExact, CallBack::polinom2_ey, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&EzExact, CallBack::polinom2_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::polinom2_hx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::polinom2_hy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HzExact, CallBack::polinom2_hz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::polinom2_ex);
    intScheme->setCallbackEyExact (CallBack::polinom2_ey);
    intScheme->setCallbackEzExact (CallBack::polinom2_ez);

    intScheme->setCallbackHxExact (CallBack::polinom2_hx);
    intScheme->setCallbackHyExact (CallBack::polinom2_hy);
    intScheme->setCallbackHzExact (CallBack::polinom2_hz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom3DiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::polinom3_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::polinom3_hy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::polinom3_ez);
    intScheme->setCallbackHyExact (CallBack::polinom3_hy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateSin1DiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::sin1_ez, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::sin1_hy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::sin1_ez);
    intScheme->setCallbackHyExact (CallBack::sin1_hy);
#endif
  }
#endif

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp1_ex_exhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp1_hy_exhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp1_ex_exhy);
    intScheme->setCallbackHyExact (CallBack::exp1_hy_exhy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp2_ex_exhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp2_hy_exhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp2_ex_exhy);
    intScheme->setCallbackHyExact (CallBack::exp2_hy_exhy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp3_ex_exhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp3_hy_exhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp3_ex_exhy);
    intScheme->setCallbackHyExact (CallBack::exp3_hy_exhy);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp1_ex_exhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HzExact, CallBack::exp1_hz_exhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp1_ex_exhz);
    intScheme->setCallbackHzExact (CallBack::exp1_hz_exhz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp2_ex_exhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HzExact, CallBack::exp2_hz_exhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp2_ex_exhz);
    intScheme->setCallbackHzExact (CallBack::exp2_hz_exhz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&ExExact, CallBack::exp3_ex_exhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HzExact, CallBack::exp3_hz_exhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackExExact (CallBack::exp3_ex_exhz);
    intScheme->setCallbackHzExact (CallBack::exp3_hz_exhz);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp1_ey_eyhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp1_hx_eyhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp1_ey_eyhx);
    intScheme->setCallbackHxExact (CallBack::exp1_hx_eyhx);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp2_ey_eyhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp2_hx_eyhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp2_ey_eyhx);
    intScheme->setCallbackHxExact (CallBack::exp2_hx_eyhx);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp3_ey_eyhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp3_hx_eyhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp3_ey_eyhx);
    intScheme->setCallbackHxExact (CallBack::exp3_hx_eyhx);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp1_ey_eyhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp1_hz_eyhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp1_ey_eyhz);
    intScheme->setCallbackHzExact (CallBack::exp1_hz_eyhz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp2_ey_eyhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp2_hz_eyhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp2_ey_eyhz);
    intScheme->setCallbackHzExact (CallBack::exp2_hz_eyhz);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHzDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EyExact, CallBack::exp3_ey_eyhz, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp3_hz_eyhz, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEyExact (CallBack::exp3_ey_eyhz);
    intScheme->setCallbackHzExact (CallBack::exp3_hz_eyhz);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp1_ez_ezhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp1_hx_ezhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp1_ez_ezhx);
    intScheme->setCallbackHxExact (CallBack::exp1_hx_ezhx);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp2_ez_ezhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp2_hx_ezhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp2_ez_ezhx);
    intScheme->setCallbackHxExact (CallBack::exp2_hx_ezhx);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHxDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp3_ez_ezhx, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HxExact, CallBack::exp3_hx_ezhx, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp3_ez_ezhx);
    intScheme->setCallbackHxExact (CallBack::exp3_hx_ezhx);
#endif
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp1_ez_ezhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp1_hy_ezhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp1_ez_ezhy);
    intScheme->setCallbackHyExact (CallBack::exp1_hy_ezhy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp2_ez_ezhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp2_hy_ezhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp2_ez_ezhy);
    intScheme->setCallbackHyExact (CallBack::exp2_hy_ezhy);
#endif
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHyDiffNorm ())
  {
#ifdef CUDA_ENABLED
    SourceCallBack tmp;
    cudaMemcpyFromSymbol (&EzExact, CallBack::exp3_ez_ezhy, sizeof(SourceCallBack));
    cudaMemcpyFromSymbol (&HyExact, CallBack::exp3_hy_ezhy, sizeof(SourceCallBack));
#else
    intScheme->setCallbackEzExact (CallBack::exp3_ez_ezhy);
    intScheme->setCallbackHyExact (CallBack::exp3_hy_ezhy);
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initMaterialFromFile (GridType gridType, Grid<TC> *grid, Grid<TC> *totalGrid)
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
    for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = grid->calculatePositionFromIndex (i);
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

  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    TC pos = grid->calculatePositionFromIndex (i);
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

  intScheme->getEps ()->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::EPS, intScheme->getEps (), totalEps);

  if (SOLVER_SETTINGS.getEpsSphere () != 1)
  {
    for (grid_coord i = 0; i < intScheme->getEps ()->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = intScheme->getEps ()->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getEps ()->getTotalPosition (pos));
      FieldValue *val = intScheme->getEps ()->getFieldValue (pos, 0);

      FieldValue epsVal = getFieldValueRealOnly (SOLVER_SETTINGS.getEpsSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getEpsSphereCenterX (),
                                             SOLVER_SETTINGS.getEpsSphereCenterY (),
                                             SOLVER_SETTINGS.getEpsSphereCenterZ (),
                                             ct1, ct2, ct3);
      *val = Approximation::approximateSphereAccurate (posAbs,
                                                       center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                       SOLVER_SETTINGS.getEpsSphereRadius () * modifier,
                                                       epsVal,
                                                       getFieldValueRealOnly (1.0));
    }
  }
  if (SOLVER_SETTINGS.getUseEpsAllNorm ())
  {
    for (grid_coord i = 0; i < intScheme->getEps ()->getSize ().calculateTotalCoord (); ++i)
    {
      FieldValue *val = intScheme->getEps ()->getFieldValue (i, 0);
      *val = getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Eps0);
    }
  }

  intScheme->getMu ()->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::MU, intScheme->getMu (), totalMu);

  if (SOLVER_SETTINGS.getMuSphere () != 1)
  {
    for (grid_coord i = 0; i < intScheme->getMu ()->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = intScheme->getMu ()->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getMuCoordFP (intScheme->getMu ()->getTotalPosition (pos));
      FieldValue *val = intScheme->getMu ()->getFieldValue (pos, 0);

      FieldValue muVal = getFieldValueRealOnly (SOLVER_SETTINGS.getMuSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getMuSphereCenterX (),
                                             SOLVER_SETTINGS.getMuSphereCenterY (),
                                             SOLVER_SETTINGS.getMuSphereCenterZ (),
                                             ct1, ct2, ct3);
      *val = Approximation::approximateSphereAccurate (posAbs,
                                                       center * modifier + TC_FP_COORD (0.5, 0.5, 0.5, ct1, ct2, ct3),
                                                       SOLVER_SETTINGS.getMuSphereRadius () * modifier,
                                                       muVal,
                                                       getFieldValueRealOnly (1.0));
    }
  }
  if (SOLVER_SETTINGS.getUseMuAllNorm ())
  {
    for (grid_coord i = 0; i < intScheme->getMu ()->getSize ().calculateTotalCoord (); ++i)
    {
      FieldValue *val = intScheme->getMu ()->getFieldValue (i, 0);
      *val = getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Mu0);
    }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    initMaterialFromFile (GridType::OMEGAPE, intScheme->getOmegaPE (), totalOmegaPE);

    if (SOLVER_SETTINGS.getOmegaPESphere () != 0)
    {
      for (grid_coord i = 0; i < intScheme->getOmegaPE ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getOmegaPE ()->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getOmegaPE ()->getTotalPosition (pos));
        FieldValue *val = intScheme->getOmegaPE ()->getFieldValue (pos, 0);

        FieldValue omegapeVal = getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPESphere () * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ());

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPESphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterZ (),
                                                ct1, ct2, ct3);
        *val = Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    SOLVER_SETTINGS.getOmegaPESphereRadius () * modifier,
                                                                    omegapeVal,
                                                                    getFieldValueRealOnly (0.0));
      }
    }

    initMaterialFromFile (GridType::OMEGAPM, intScheme->getOmegaPM (), totalOmegaPM);

    if (SOLVER_SETTINGS.getOmegaPMSphere () != 0)
    {
      for (grid_coord i = 0; i < intScheme->getOmegaPM ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getOmegaPM ()->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (intScheme->getOmegaPM ()->getTotalPosition (pos));
        FieldValue *val = intScheme->getOmegaPM ()->getFieldValue (pos, 0);

        FieldValue omegapmVal = getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPMSphere () * 2 * PhysicsConst::Pi * intScheme->getSourceFrequency ());

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPMSphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterZ (),
                                                ct1, ct2, ct3);
        *val = Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    SOLVER_SETTINGS.getOmegaPMSphereRadius () * modifier,
                                                                    omegapmVal,
                                                                    getFieldValueRealOnly (0.0));
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

        dumper[type]->dumpGrid (totalEps, startEps, endEps, 0, 0);
        dumper[type]->dumpGrid (totalMu, startMu, endMu, 0, 0);

        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          dumper[type]->dumpGrid (totalOmegaPE, startOmegaPE, endOmegaPE, 0, 0);
          dumper[type]->dumpGrid (totalOmegaPM, startOmegaPM, endOmegaPM, 0, 0);
          dumper[type]->dumpGrid (totalGammaE, startGammaE, endGammaE, 0, 0);
          dumper[type]->dumpGrid (totalGammaM, startGammaM, endGammaM, 0, 0);
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
      for (grid_coord i = 0; i < intScheme->getEx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEx ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getEy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEy ()->calculatePositionFromIndex (i);
        
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
      for (grid_coord i = 0; i < intScheme->getEz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEz ()->calculatePositionFromIndex (i);
        
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
      for (grid_coord i = 0; i < intScheme->getHx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHx ()->calculatePositionFromIndex (i);
        
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
      for (grid_coord i = 0; i < intScheme->getHy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHy ()->calculatePositionFromIndex (i);
        
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
      for (grid_coord i = 0; i < intScheme->getHz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHz ()->calculatePositionFromIndex (i);
        
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
      for (grid_coord i = 0; i < intScheme->getEx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEx ()->calculatePositionFromIndex (i);

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
        intScheme->getCcPMLEx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
    
    if (intScheme->getDoNeedEy ())
    {
      for (grid_coord i = 0; i < intScheme->getEy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEy ()->calculatePositionFromIndex (i);

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
        intScheme->getCcPMLEy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
    
    if (intScheme->getDoNeedEz ())
    {
      for (grid_coord i = 0; i < intScheme->getEz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEz ()->calculatePositionFromIndex (i);

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
        intScheme->getCcPMLEz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
    
    if (intScheme->getDoNeedHx ())
    {
      for (grid_coord i = 0; i < intScheme->getHx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHx ()->calculatePositionFromIndex (i);

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
        intScheme->getDcPMLHx ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
    
    if (intScheme->getDoNeedHy ())
    {
      for (grid_coord i = 0; i < intScheme->getHy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHy ()->calculatePositionFromIndex (i);

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
        intScheme->getDcPMLHy ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
    
    if (intScheme->getDoNeedHz ())
    {
      for (grid_coord i = 0; i < intScheme->getHz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHz ()->calculatePositionFromIndex (i);

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
        intScheme->getDcPMLHz ()->setFieldValue (FIELDVALUE (Cb, FPValue (0)), pos, 0);
      }
    }
  }

  if (SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ()
      && SOLVER_SETTINGS.getDoUsePML ()
      && SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    if (intScheme->getDoNeedEx ())
    {
      for (grid_coord i = 0; i < intScheme->getEx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEx ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getEy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEy ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getEz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getEz ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getHx ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHx ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getHy ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHy ()->calculatePositionFromIndex (i);

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
      for (grid_coord i = 0; i < intScheme->getHz ()->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = intScheme->getHz ()->calculatePositionFromIndex (i);

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
//
// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// FPValue
// Scheme<Type, TCoord, layout_type>::Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *curEx, Grid<TC> *curEy, Grid<TC> *curEz,
//                        Grid<TC> *curHx, Grid<TC> *curHy, Grid<TC> *curHz)
// {
// #ifdef COMPLEX_FIELD_VALUES
//   FPValue k = 2 * PhysicsConst::Pi / sourceWaveLength; // TODO: check numerical here
//
//   NPair N = ntffN (angleTeta, anglePhi, curEz, curHx, curHy, curHz);
//   NPair L = ntffL (angleTeta, anglePhi, curEx, curEy, curEz);
//
//   int processId = 0;
//
//   if (useParallel)
//   {
// #ifdef PARALLEL_GRID
//     processId = ParallelGrid::getParallelCore ()->getProcessId ();
//
//     FieldValue tmpArray[4];
//     FieldValue tmpArrayRes[4];
//     const int count = 4;
//
//     tmpArray[0] = N.nTeta;
//     tmpArray[1] = N.nPhi;
//     tmpArray[2] = L.nTeta;
//     tmpArray[3] = L.nPhi;
//
//     // gather all sum_teta and sum_phi on 0 node
//     MPI_Reduce (tmpArray, tmpArrayRes, count, MPI_FPVALUE, MPI_SUM, 0, ParallelGrid::getParallelCore ()->getCommunicator ());
//
//     if (processId == 0)
//     {
//       N.nTeta = FieldValue (tmpArrayRes[0]);
//       N.nPhi = FieldValue (tmpArrayRes[1]);
//
//       L.nTeta = FieldValue (tmpArrayRes[2]);
//       L.nPhi = FieldValue (tmpArrayRes[3]);
//     }
// #else
//     ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
// #endif
//   }
//
//   if (processId == 0)
//   {
//     FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);
//
//     FieldValue first = -L.nPhi + N.nTeta * n0;
//     FieldValue second = -L.nTeta - N.nPhi * n0;
//
//     FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
//     FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());
//
//     return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
//   }
//   else
//   {
//     return 0.0;
//   }
// #else
//   ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
// #endif
// }
//
// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// FPValue
// Scheme<Type, TCoord, layout_type>::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
// {
//   return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0);
// }

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::makeGridScattered (Grid<TC> *grid, GridType gridType)
{
  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    FieldValue *val = grid->getFieldValue (i, 1);

    TC pos = grid->calculatePositionFromIndex (i);
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
          totalEx = new Grid<TC> (yeeLayout->getExSize (), 0, intScheme->getEx ()->getCountStoredSteps (), "TotalEx");
          totalEx->copy (intScheme->getEx ());
        }
        if (intScheme->getDoNeedEy ())
        {
          totalEy = new Grid<TC> (yeeLayout->getEySize (), 0, intScheme->getEy ()->getCountStoredSteps (), "TotalEy");
          totalEy->copy (intScheme->getEy ());
        }
        if (intScheme->getDoNeedEz ())
        {
          totalEz = new Grid<TC> (yeeLayout->getEzSize (), 0, intScheme->getEz ()->getCountStoredSteps (), "TotalEz");
          totalEz->copy (intScheme->getEz ());
        }

        if (intScheme->getDoNeedHx ())
        {
          totalHx = new Grid<TC> (yeeLayout->getHxSize (), 0, intScheme->getHx ()->getCountStoredSteps (), "TotalHx");
          totalHx->copy (intScheme->getHx ());
        }
        if (intScheme->getDoNeedHy ())
        {
          totalHy = new Grid<TC> (yeeLayout->getHySize (), 0, intScheme->getHy ()->getCountStoredSteps (), "TotalHy");
          totalHy->copy (intScheme->getHy ());
        }
        if (intScheme->getDoNeedHz ())
        {
          totalHz = new Grid<TC> (yeeLayout->getHzSize (), 0, intScheme->getHz ()->getCountStoredSteps (), "TotalHz");
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
  if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
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

    if (intScheme->getDoNeedEx ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEx (), zero, intScheme->getEx ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalEx, startEx, endEx, t, -1);
      }
    }

    if (intScheme->getDoNeedEy ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEy (), zero, intScheme->getEy ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalEy, startEy, endEy, t, -1);
      }
    }

    if (intScheme->getDoNeedEz ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getEz (), zero, intScheme->getEz ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalEz, startEz, endEz, t, -1);
      }
    }

    if (intScheme->getDoNeedHx ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHx (), zero, intScheme->getHx ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalHx, startHx, endHx, t, -1);
      }
    }

    if (intScheme->getDoNeedHy ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHy (), zero, intScheme->getHy ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalHy, startHy, endHy, t, -1);
      }
    }

    if (intScheme->getDoNeedHz ())
    {
      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (intScheme->getHz (), zero, intScheme->getHz ()->getSize (), t, -1);
      }
      else
      {
        dumper[type]->dumpGrid (totalHz, startHz, endHz, t, -1);
      }
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFEInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->dumpGrid (intScheme->getEInc (), GRID_COORDINATE_1D (0, CoordinateType::X), intScheme->getEInc ()->getSize (), t, -1);
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFHInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->dumpGrid (intScheme->getHInc (), GRID_COORDINATE_1D (0, CoordinateType::X), intScheme->getHInc ()->getSize (), t, -1);
    }
  }
}
//
// template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
// void
// Scheme<Type, TCoord, layout_type>::saveNTFF (bool isReverse, time_step t)
// {
//   int processId = 0;
//
//   if (useParallel)
//   {
// #ifdef PARALLEL_GRID
//     processId = ParallelGrid::getParallelCore ()->getProcessId ();
// #else
//     ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
// #endif
//   }
//
//   std::ofstream outfile;
//   std::ostream *outs;
//   const char *strName;
//   FPValue start;
//   FPValue end;
//   FPValue step;
//
//   if (isReverse)
//   {
//     strName = "Reverse diagram";
//     start = yeeLayout->getIncidentWaveAngle2 ();
//     end = yeeLayout->getIncidentWaveAngle2 ();
//     step = 1.0;
//   }
//   else
//   {
//     strName = "Forward diagram";
//     start = 0.0;
//     end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
//     step = PhysicsConst::Pi * SOLVER_SETTINGS.getAngleStepNTFF () / 180;
//   }
//
//   if (processId == 0)
//   {
//     if (SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
//     {
//       outs = &std::cout;
//     }
//     else
//     {
//       outfile.open (SOLVER_SETTINGS.getFileNameNTFF ().c_str ());
//       outs = &outfile;
//     }
//     (*outs) << strName << std::endl << std::endl;
//   }
//
//   for (FPValue angle = start; angle <= end; angle += step)
//   {
//     FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
//                                  angle,
//                                  internalScheme.Ex,
//                                  internalScheme.Ey,
//                                  internalScheme.Ez,
//                                  internalScheme.Hx,
//                                  internalScheme.Hy,
//                                  internalScheme.Hz) / Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);
//
//     if (processId == 0)
//     {
//       (*outs) << "timestep = "
//               << t
//               << ", incident wave angle=("
//               << yeeLayout->getIncidentWaveAngle1 () << ","
//               << yeeLayout->getIncidentWaveAngle2 () << ","
//               << yeeLayout->getIncidentWaveAngle3 () << ","
//               << "), angle NTFF = "
//               << angle
//               << ", NTFF value = "
//               << val
//               << std::endl;
//     }
//   }
//
//   if (processId == 0)
//   {
//     if (!SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
//     {
//       outfile.close ();
//     }
//   }
// }

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

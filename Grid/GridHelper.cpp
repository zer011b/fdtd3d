#ifdef PARALLEL_GRID
#ifdef PRINT_MESSAGE

// Names of buffers of parallel grid for debug purposes.
const char* BufferPositionNames[] =
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "LEFT",
  "RIGHT",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "UP",
  "DOWN",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "FRONT",
  "BACK",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "LEFT_UP",
  "LEFT_DOWN",
  "RIGHT_UP",
  "RIGHT_DOWN",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "LEFT_FRONT",
  "LEFT_BACK",
  "RIGHT_FRONT",
  "RIGHT_BACK",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "UP_FRONT",
  "UP_BACK",
  "DOWN_FRONT",
  "DOWN_BACK",
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  "LEFT_UP_FRONT",
  "LEFT_UP_BACK",
  "LEFT_DOWN_FRONT",
  "LEFT_DOWN_BACK",
  "RIGHT_UP_FRONT",
  "RIGHT_UP_BACK",
  "RIGHT_DOWN_FRONT",
  "RIGHT_DOWN_BACK",
#endif

/*
 * Overall number of buffers for current dimension.
 */
  BUFFER_COUNT
};

#endif /* PRINT_MESSAGE */
#endif /* PARALLEL_GRID */

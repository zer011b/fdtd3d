#ifdef PARALLEL_GRID
#ifdef PRINT_MESSAGE

// Names of buffers of parallel grid for debug purposes.
const char* BufferPositionNames[] =
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D
  "LEFT",
  "RIGHT"
#endif /* PARALLEL_BUFFER_DIMENSION_1D */
#ifdef PARALLEL_BUFFER_DIMENSION_2D
  "LEFT",
  "RIGHT",
  "UP",
  "DOWN",
  "LEFT_UP",
  "LEFT_DOWN",
  "RIGHT_UP",
  "RIGHT_DOWN",
#endif /* PARALLEL_BUFFER_DIMENSION_2D */
#ifdef PARALLEL_BUFFER_DIMENSION_3D
  "LEFT",
  "RIGHT",
  "UP",
  "DOWN",
  "FRONT",
  "BACK",
  "LEFT_FRONT",
  "LEFT_BACK",
  "LEFT_UP",
  "LEFT_DOWN",
  "RIGHT_FRONT",
  "RIGHT_BACK",
  "RIGHT_UP",
  "RIGHT_DOWN",
  "UP_FRONT",
  "UP_BACK",
  "DOWN_FRONT",
  "DOWN_BACK",
  "LEFT_UP_FRONT",
  "LEFT_UP_BACK",
  "LEFT_DOWN_FRONT",
  "LEFT_DOWN_BACK",
  "RIGHT_UP_FRONT",
  "RIGHT_UP_BACK",
  "RIGHT_DOWN_FRONT",
  "RIGHT_DOWN_BACK",
#endif /* PARALLEL_BUFFER_DIMENSION_3D */
};

#endif /* PRINT_MESSAGE */
#endif /* PARALLEL_GRID */

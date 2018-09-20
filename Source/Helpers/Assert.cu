#define CUDA_SOURCES

#include "Assert.h"

#include <cstdlib>

__device__
void cuda_program_fail ()
{
  printf ("Error: cuda_program_fail!!\n");
}

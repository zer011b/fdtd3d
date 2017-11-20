#include "ParallelYeeGridLayout.h"

#ifdef PARALLEL_GRID

template <uint8_t layout_type>
const bool ParallelYeeGridLayout::isParallel = true;

#endif /* PARALLEL_GRID */

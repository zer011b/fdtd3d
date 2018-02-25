#include "ParallelYeeGridLayout.h"

#ifdef PARALLEL_GRID

template <SchemeType Type, uint8_t layout_type>
const bool ParallelYeeGridLayout<Type, layout_type>::isParallel = true;

#endif /* PARALLEL_GRID */

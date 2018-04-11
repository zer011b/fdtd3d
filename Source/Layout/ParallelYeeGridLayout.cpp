#include "ParallelYeeGridLayout.h"

#ifdef PARALLEL_GRID

template <SchemeType_t Type, LayoutType layout_type>
const bool ParallelYeeGridLayout<Type, layout_type>::isParallel = true;

#endif /* PARALLEL_GRID */

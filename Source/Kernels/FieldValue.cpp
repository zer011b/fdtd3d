#include "FieldValue.h"

#include <cmath>

template<>
CUDA_DEVICE CUDA_HOST
FPValue exponent (FPValue arg)
{
  return exp (arg);
}

#ifdef COMPLEX_FIELD_VALUES
template<>
CUDA_DEVICE CUDA_HOST
CComplex<FPValue> exponent (CComplex<FPValue> arg)
{
  return arg.exp ();
}
#endif /* COMPLEX_FIELD_VALUES */

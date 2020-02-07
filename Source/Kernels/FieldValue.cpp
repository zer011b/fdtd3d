#include "FieldValue.h"

#include <cmath>

template<>
CUDA_DEVICE CUDA_HOST
FPValue exponent (FPValue arg)
{
  return exp (arg);
}

#ifdef COMPLEX_FIELD_VALUES
#ifndef STD_COMPLEX
template<>
CUDA_DEVICE CUDA_HOST
CComplex<FPValue> exponent (CComplex<FPValue> arg)
{
  return arg.exp ();
}
#else /* !STD_COMPLEX */
template<>
CUDA_DEVICE CUDA_HOST
std::complex<FPValue> exponent (std::complex<FPValue> arg)
{
  return std::exp (arg);
}
#endif /* STD_COMPLEX */

#endif /* COMPLEX_FIELD_VALUES */

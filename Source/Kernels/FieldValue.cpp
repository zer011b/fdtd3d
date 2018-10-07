#include "FieldValue.h"

#include <cmath>

CUDA_DEVICE CUDA_HOST
FieldValue getFieldValueRealOnly (FPValue real)
{
#ifdef COMPLEX_FIELD_VALUES
  return FieldValue (real, 0.0);
#else /* COMPLEX_FIELD_VALUES*/
  return FieldValue (real);
#endif /* !COMPLEX_FIELD_VALUES */
}

CUDA_DEVICE CUDA_HOST
FPValue getRealOnlyFromFieldValue (const FieldValue &val)
{
#ifdef COMPLEX_FIELD_VALUES
  return val.real ();
#else /* COMPLEX_FIELD_VALUES*/
  return val;
#endif /* !COMPLEX_FIELD_VALUES */
}

template<>
CUDA_DEVICE CUDA_HOST
FPValue exponent (FPValue arg)
{
  return exp (arg);
}

#ifdef COMPLEX_FIELD_VALUES
#ifndef STD_COMPEX
template<>
CUDA_DEVICE CUDA_HOST
CComplex<FPValue> exponent (CComplex<FPValue> arg)
{
  return arg.exp ();
}
#else /* !STD_COMPEX */
template<>
CUDA_DEVICE CUDA_HOST
std::complex<FPValue> exponent (std::complex<FPValue> arg)
{
  return std::exp (arg);
}
#endif /* STD_COMPEX */

#endif /* COMPLEX_FIELD_VALUES */

#ifndef FIELD_VALUES_H
#define FIELD_VALUES_H

/**
 * Type of floating point values
 */
#ifdef FLOAT_VALUES
typedef float FPValue;
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
typedef double FPValue;
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
typedef long double FPValue;
#endif /* LONG_DOUBLE_VALUES */

/**
 * Type of field values
 */
#ifdef COMPLEX_FIELD_VALUES
#include <complex>
typedef std::complex<FPValue> FieldValue;
#else /* COMPLEX_FIELD_VALUES */
typedef FPValue FieldValue;
#endif /* !COMPLEX_FIELD_VALUES */

#ifdef CXX11_ENABLED
#include <cstdint>
#define CXX11_OVERRIDE override
#define CXX11_FINAL final
#define CXX11_OVERRIDE_FINAL CXX11_OVERRIDE CXX11_FINAL
#else /* CXX11_ENABLED */
#include <stdint.h>
#define CXX11_OVERRIDE
#define CXX11_FINAL
#define CXX11_OVERRIDE_FINAL
#endif /* !CXX11_ENABLED */

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else /* __CUDACC__ */
#define CUDA_DEVICE
#define CUDA_HOST
#endif /* !__CUDACC__ */

/**
 * Type of one-dimensional coordinate.
 */
typedef uint32_t grid_coord;

/**
 * Type of iterator through one-dimensional array
 */
typedef uint64_t grid_iter;

/**
 * Type of timestep
 */
typedef uint32_t time_step;

#endif /* FIELD_VALUES_H */

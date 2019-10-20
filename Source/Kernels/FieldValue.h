#ifndef FIELD_VALUES_H
#define FIELD_VALUES_H

#include "Assert.h"

/**
 * Type of floating point values
 */
#ifdef FLOAT_VALUES
typedef float FPValue;

#ifdef COMPLEX_FIELD_VALUES
#define MPI_FPVALUE MPI_COMPLEX
#else /* COMPLEX_FIELD_VALUES */
#define MPI_FPVALUE MPI_FLOAT
#endif /* !COMPLEX_FIELD_VALUES */

#define FP_MOD "%f"
#define FP_MOD_ACC "%.20f"
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
typedef double FPValue;

#ifdef COMPLEX_FIELD_VALUES
#define MPI_FPVALUE MPI_DOUBLE_COMPLEX
#else /* COMPLEX_FIELD_VALUES */
#define MPI_FPVALUE MPI_DOUBLE
#endif /* !COMPLEX_FIELD_VALUES */

#define FP_MOD "%f"
#define FP_MOD_ACC "%.20f"
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
typedef long double FPValue;

#ifdef COMPLEX_FIELD_VALUES
#define MPI_FPVALUE MPI_LONG_DOUBLE_COMPLEX
#else /* COMPLEX_FIELD_VALUES */
#define MPI_FPVALUE MPI_LONG_DOUBLE
#endif /* !COMPLEX_FIELD_VALUES */

#define FP_MOD "%Lf"
#define FP_MOD_ACC "%.20Lf"
#endif /* LONG_DOUBLE_VALUES */

/**
 * Type of field values
 */
#ifdef COMPLEX_FIELD_VALUES
#ifdef STD_COMPLEX
#include <complex>
typedef std::complex<FPValue> FieldValue;
#else /* STD_COMPLEX */
#include "Complex.h"
typedef CComplex<FPValue> FieldValue;
#endif /* !STD_COMPLEX */
#define FIELDVALUE(real,imag) FieldValue(real,imag)
#else /* COMPLEX_FIELD_VALUES */
typedef FPValue FieldValue;
#define FIELDVALUE(real,imag) FieldValue(real)
#endif /* !COMPLEX_FIELD_VALUES */

#ifdef CXX11_ENABLED
#include <cstdint>
#include <cinttypes>
#define CXX11_OVERRIDE override
#define CXX11_FINAL final
#define CXX11_OVERRIDE_FINAL CXX11_OVERRIDE CXX11_FINAL
#define NULLPTR nullptr
#else /* CXX11_ENABLED */
#include <stdint.h>
#include <inttypes.h>
#define CXX11_OVERRIDE
#define CXX11_FINAL
#define CXX11_OVERRIDE_FINAL
#define NULLPTR NULL
#endif /* !CXX11_ENABLED */

/**
 * Type of one-dimensional coordinate.
 */
#ifndef LARGE_COORDINATES
typedef int32_t grid_coord;
#define MPI_COORD MPI_INT
//#define MAX_COORD (2048*1048576) // 2^31, coord should be less than this
#else /* !LARGE_COORDINATES */
typedef int64_t grid_coord;
#define MPI_COORD MPI_LONG_LONG_INT
//#define MAX_COORD (8*1048576*1048576*1048576) // 2^63, coord should be less than this
#endif /* LARGE_COORDINATES */

/**
 * Type of timestep
 */
typedef uint32_t time_step;

extern CUDA_DEVICE CUDA_HOST FieldValue getFieldValueRealOnly (FPValue);
extern CUDA_DEVICE CUDA_HOST FPValue getRealOnlyFromFieldValue (const FieldValue &);

#ifdef FLOAT_VALUES
#define FPEXACT_ACCURACY (FPValue(0.000001))
#endif
#ifdef DOUBLE_VALUES
#define FPEXACT_ACCURACY (FPValue(0.0000001))
#endif
#ifdef LONG_DOUBLE_VALUES
#define FPEXACT_ACCURACY (FPValue(0.0000001))
#endif

#define IS_FP_EXACT(a,b) \
  (((a) > (b) ? (a) - (b) : (b) - (a)) < FPEXACT_ACCURACY)

template<class T>
CUDA_DEVICE CUDA_HOST
T exponent (T arg);

#endif /* FIELD_VALUES_H */

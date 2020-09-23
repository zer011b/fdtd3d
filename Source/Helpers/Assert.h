#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>
#include <assert.h>

/*
 * CUDA notes.
 *
 * 1. CUDA_SOURCES should be defined for all cuda sources (.cu files) in order to include GPU related stuff.
 *    All the stuff that should be defined for Cuda, but both for CPU and GPU code, should be under #ifdef CUDA_SOURCES
 * 2. __CUDA_ARCH__ allows to define stuff separately for CPU and GPU even in the same .cu file (because the file will
 *    be preprocessed for CPU and GPU separately)
 */

#ifndef __CUDA_ARCH__
extern void program_fail ();
#endif /* !__CUDA_ARCH__ */

#ifdef __CUDA_ARCH__
#define PROGRAM_FAIL assert(0)
#define SOLVER_SETTINGS (*cudaSolverSettings)
//#define PROGRAM_FAIL_EXIT *retval = CUDA_ERROR; return;
#define PROGRAM_FAIL_EXIT
#define PROGRAM_OK_EXIT *retval = CUDA_OK; return;
#else /* __CUDA_ARCH__ */
#define PROGRAM_FAIL program_fail()
#define SOLVER_SETTINGS solverSettings
#define PROGRAM_FAIL_EXIT
#define PROGRAM_OK_EXIT
#endif /* !__CUDA_ARCH__ */

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else /* __CUDACC__ */
#define CUDA_DEVICE
#define CUDA_HOST
#endif /* !__CUDACC__ */

/*
 * Printf used for logging
 */
#if PRINT_MESSAGE
#define DPRINTF(logLevel, ...) \
  { \
    if (SOLVER_SETTINGS.getLogLevel () >= logLevel) \
    { \
      printf (__VA_ARGS__); \
    } \
  }
#else /* PRINT_MESSAGE */
#define DPRINTF(...)
#endif /* !PRINT_MESSAGE */

#define USED(var) (void)var

#ifdef ENABLE_ASSERTS
/*
 * Indicates program point, which should not be reached.
 */
#define UNREACHABLE \
{ \
  DPRINTF (LOG_LEVEL_NONE, "Unreachable executed at %s:%d.\n", __FILE__, __LINE__); \
  PROGRAM_FAIL; \
  PROGRAM_FAIL_EXIT \
}

/*
 * Unconditional assert with message.
 */
#define ASSERT_MESSAGE(x) \
{ \
  DPRINTF (LOG_LEVEL_NONE, "Assert '%s' at %s:%d.\n", x, __FILE__, __LINE__); \
  PROGRAM_FAIL; \
  PROGRAM_FAIL_EXIT \
}

/*
 * Conditional assert with default message.
 */
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    DPRINTF (LOG_LEVEL_NONE, "Assert at %s:%d.\n", __FILE__, __LINE__); \
    PROGRAM_FAIL; \
    PROGRAM_FAIL_EXIT \
  } \
}
#else /* ENABLE_ASSERTS */
#define UNREACHABLE
#define ASSERT_MESSAGE(x)
#define ASSERT(x)
#endif /* !ENABLE_ASSERTS */

#define ALWAYS_ASSERT(x) \
{ \
  if (!(x)) \
  { \
    DPRINTF (LOG_LEVEL_NONE, "Assert at %s:%d.\n", __FILE__, __LINE__); \
    PROGRAM_FAIL; \
    PROGRAM_FAIL_EXIT \
  } \
}

#define ALWAYS_ASSERT_MESSAGE(x) \
{ \
  DPRINTF (LOG_LEVEL_NONE, "Assert '%s' at %s:%d.\n", x, __FILE__, __LINE__); \
  PROGRAM_FAIL; \
  PROGRAM_FAIL_EXIT \
}

/*
 * Enum class for c++11 and not c++11 builds
 */
#ifdef CXX11_ENABLED
#define ENUM_CLASS(name, type, ...) \
  enum class name : type \
  { \
    __VA_ARGS__ \
  };
#else /* CXX11_ENABLED */
#define ENUM_CLASS(name, type, ...) \
  class name \
  { \
    public: \
    \
    enum Temp { __VA_ARGS__ }; \
    \
    CUDA_DEVICE CUDA_HOST name (Temp new_val) : temp (new_val) {} \
    CUDA_DEVICE CUDA_HOST name () {} \
    \
    CUDA_DEVICE CUDA_HOST operator type () const { return temp; } \
    CUDA_DEVICE CUDA_HOST bool operator < (name x) const { return temp < x.temp; } \
    CUDA_DEVICE CUDA_HOST bool operator > (name x) const { return temp > x.temp; } \
    CUDA_DEVICE CUDA_HOST bool operator == (name x) const { return temp == x.temp; } \
    CUDA_DEVICE CUDA_HOST bool operator == (Temp t) const { return temp == t; } \
    \
  private: \
    Temp temp; \
  };
#endif /* !CXX11_ENABLED */

/*
 * String to number
 */
#ifdef CXX11_ENABLED
#define STOI(str) std::stoi (str)
#define STOF(str) std::stof (str)
#else /* CXX11_ENABLED */
#define STOI(str) atoi (str)
#define STOF(str) atof (str)
#endif /* !CXX11_ENABLED */

#define EXIT_OK 0x0
#define EXIT_BREAK_ARG_PARSING 0x1
#define EXIT_ERROR 0xa
#define EXIT_UNKNOWN_OPTION 0xb

/*
 * This should correspond to MPI_DOUBLE!
 */
typedef double DOUBLE;

/**
 * Macro for square
 */
#define SQR(x) ((x) * (x))

/**
 * Macro for cube
 */
#define CUBE(x) (SQR(x) * (x))

#ifdef DEBUG_INFO
#define TCOORD(x,y,z,ct1,ct2,ct3) TCoord(x,y,z,ct1,ct2,ct3)
#define TC_COORD(x,y,z,ct1,ct2,ct3) TC(x,y,z,ct1,ct2,ct3)
#define GRID_COORDINATE_1D(x,ct1) GridCoordinate1D(x,ct1)
#define GRID_COORDINATE_2D(x,y,ct1,ct2) GridCoordinate2D(x,y,ct1,ct2)
#define GRID_COORDINATE_3D(x,y,z,ct1,ct2,ct3) GridCoordinate3D(x,y,z,ct1,ct2,ct3)
#define TC_FP_COORD(x,y,z,ct1,ct2,ct3) TCFP(x,y,z,ct1,ct2,ct3)
#define GRID_COORDINATE_FP_1D(x,ct1) GridCoordinateFP1D(x,ct1)
#define GRID_COORDINATE_FP_2D(x,y,ct1,ct2) GridCoordinateFP2D(x,y,ct1,ct2)
#define GRID_COORDINATE_FP_3D(x,y,z,ct1,ct2,ct3) GridCoordinateFP3D(x,y,z,ct1,ct2,ct3)
#else /* DEBUG_INFO */
#define TCOORD(x,y,z,ct1,ct2,ct3) TCoord(x,y,z)
#define TC_COORD(x,y,z,ct1,ct2,ct3) TC(x,y,z)
#define GRID_COORDINATE_1D(x,ct1) GridCoordinate1D(x)
#define GRID_COORDINATE_2D(x,y,ct1,ct2) GridCoordinate2D(x,y)
#define GRID_COORDINATE_3D(x,y,z,ct1,ct2,ct3) GridCoordinate3D(x,y,z)
#define TC_FP_COORD(x,y,z,ct1,ct2,ct3) TCFP(x,y,z)
#define GRID_COORDINATE_FP_1D(x,ct1) GridCoordinateFP1D(x)
#define GRID_COORDINATE_FP_2D(x,y,ct1,ct2) GridCoordinateFP2D(x,y)
#define GRID_COORDINATE_FP_3D(x,y,z,ct1,ct2,ct3) GridCoordinateFP3D(x,y,z)
#endif /* !DEBUG_INFO */

#include "CudaInclude.h"

#endif /* ASSERT_H */

/*
 * Copyright (C) 2018 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * Unit test for basic operations with Grid
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "PAssert.h"
#include "Complex.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

#define FPEXACT_ACCURACY (T(0.0001))
#define IS_FP_EXACT(a,b) \
  (((a) > (b) ? (a) - (b) : (b) - (a)) < FPEXACT_ACCURACY)

template<class T>
void test (T v1, T v2, T v3, T v4, T v5)
{
  /*
   * Constructor
   */
  CComplex<T> val1 (v1, v2);
  CComplex<T> val2 (v3);
  CComplex<T> val3;

  /*
   * Interfaces
   */
  assert (val3.real () == T (0) && val3.imag () == T (0));
  assert (val2.real () == v3 && val2.imag () == T (0));
  assert (val1.real () == v1 && val1.imag () == v2);

  assert (val3.abs () == T (0) && val3.norm () == T (0));
  assert (val2.abs () == v3 && val2.norm () == T (SQR (v3)));
  assert (val1.abs () == T (sqrt (SQR (v1) + SQR (v2)))
          && val1.norm () == T (SQR (v1) + SQR (v2)));

  CComplex<T> val4 (val1);
  assert (val4.real () == val1.real () && val4.imag () == val1.imag ());

  assert (val3.exp ().real () == T (1) && val3.exp ().imag () == T (0));

  CComplex<T> val222 (T (2), T (3));
  assert (val222.exp ().real () == T (::exp (T (2)) * cos (T (3)))
          && val222.exp ().imag () == T (::exp (T (2)) * sin (T (3))));

  /*
   * Operators
   */
  assert (val1 == val4);

  CComplex<T> val5 (v3, v4);

  assert (val1 != val5);

  CComplex<T> val6 = val1 + val5;
  assert (val6.real () == v1 + v3 && val6.imag () == v2 + v4);

  CComplex<T> val7 = val1 - val5;
  assert (val7.real () == v1 - v3 && val7.imag () == v2 - v4);

  CComplex<T> val8 = val1 * v5;
  assert (val8.real () == v1 * v5 && val8.imag () == v2 * v5);

  CComplex<T> val9 = val1 * val5;
  assert (val9.real () == v1 * v3 - v2 * v4
          && val9.imag () == v1 * v4 + v2 * v3);

  CComplex<T> val10 = val1 / v5;
  assert (val10.real () == v1 / v5 && val10.imag () == v2 / v5);

  CComplex<T> val11 = val1 / val5;
  T d = SQR (v3) + SQR (v4);
  assert (val11.real () == T ((v1 * v3 + v2 * v4) / d)
          && val11.imag () == T ((v3 * v2 - v1 * v4) / d));

  CComplex<T> val12 = -val1;
  assert (val12.real () == -v1 && val12.imag () == -v2);

  CComplex<T> val13 = val1;
  val13 -= val5;
  assert (val13.real () == v1 - v3 && val13.imag () == v2 - v4);

  CComplex<T> val14 = val1;
  val14 += val5;
  assert (val14.real () == v1 + v3 && val14.imag () == v2 + v4);

  /*
   * Comparison with std::complex
   */
  std::complex<T> _val1 (v1, v2);
  std::complex<T> _val2 (v3, v4);
  std::complex<T> _val3 = _val1 + _val2;
  std::complex<T> _val4 = _val1 - _val2;
  std::complex<T> _val5 = _val1 * _val2;
  std::complex<T> _val6 = _val1 / _val2;
  std::complex<T> _val7 = _val1 * v5;
  std::complex<T> _val8 = _val1 / v5;

  assert (IS_FP_EXACT (_val1.real (), val1.real ()) && IS_FP_EXACT (_val1.imag (), val1.imag ()));
  assert (IS_FP_EXACT (_val2.real (), val5.real ()) && IS_FP_EXACT (_val2.imag (), val5.imag ()));
  assert (IS_FP_EXACT (_val3.real (), val6.real ()) && IS_FP_EXACT (_val3.imag (), val6.imag ()));
  assert (IS_FP_EXACT (_val4.real (), val7.real ()) && IS_FP_EXACT (_val4.imag (), val7.imag ()));
  assert (IS_FP_EXACT (_val5.real (), val9.real ()) && IS_FP_EXACT (_val5.imag (), val9.imag ()));
  assert (IS_FP_EXACT (_val6.real (), val11.real ()) && IS_FP_EXACT (_val6.imag (), val11.imag ()));
  assert (IS_FP_EXACT (_val7.real (), val8.real ()) && IS_FP_EXACT (_val7.imag (), val8.imag ()));
  assert (IS_FP_EXACT (_val8.real (), val10.real ()) && IS_FP_EXACT (_val8.imag (), val10.imag ()));

  assert (IS_FP_EXACT (std::abs (_val1), val1.abs ()));
  assert (IS_FP_EXACT (std::abs (_val2), val5.abs ()));
  assert (IS_FP_EXACT (std::abs (_val3), val6.abs ()));
  assert (IS_FP_EXACT (std::abs (_val4), val7.abs ()));
  assert (IS_FP_EXACT (std::abs (_val5), val9.abs ()));
  assert (IS_FP_EXACT (std::abs (_val6), val11.abs ()));
  assert (IS_FP_EXACT (std::abs (_val7), val8.abs ()));
  assert (IS_FP_EXACT (std::abs (_val8), val10.abs ()));

  std::complex<T> _val222 (T (2), T (3));
  assert (std::exp (_val222).real () == val222.exp ().real ()
          && std::exp (_val222).imag () == val222.exp ().imag ());
}

int main (int argc, char** argv)
{
  /*
   * Complex values with floating point val
   */
  test<float> (35.32, 19213.1, 1239.123, 0.12, 0.1222);
  test<double> (35.32, 19213.1, 1239.123, 0.12, 0.1222);

  return 0;
} /* main */

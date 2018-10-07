/*
 * Unit test for basic operations with Grid
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "Assert.h"
#include "Complex.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

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
  std::complex<T> _val2 (v3);
  std::complex<T> _val3;

  assert (_val1.real () == val1.real () && _val1.imag () == val1.imag ());
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

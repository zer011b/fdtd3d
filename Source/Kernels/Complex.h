#ifndef COMPLEX_H
#define COMPLEX_H

#include "Assert.h"

#include <cmath>

/**
 * Complex values class (ideally, should fully match <complex>, see http://www.cplusplus.com/reference/complex/).
 * This is required for CUDA.
 */
template<class T>
class CComplex
{
private:

  /**
   * Real part
   */
  T re;

  /**
   * Imaginary part
   */
  T im;

public:

  /**
   * Constructor
   */
  CUDA_DEVICE CUDA_HOST
  CComplex (const T & real = T (), /**< real part */
           const T & imag = T ()) /**< imaginary part */
  : re (real)
  , im (imag)
  {
  } /* CComplex::CComplex */

  /**
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST
  CComplex (const CComplex<T> & x) /**< complex value to copy */
  : re (x.re)
  , im (x.im)
  {
  } /* CComplex::CComplex */

  /**
   * Conversion constructor
   */
  template<class U>
  CUDA_DEVICE CUDA_HOST
  CComplex (const CComplex<U> & x) /**< complex value to convert */
  : re (x.re)
  , im (x.im)
  {
  } /* CComplex::CComplex */

  /**
   * Destructor
   */
  CUDA_DEVICE CUDA_HOST
  ~CComplex ()
  {
  } /* CComplex::~CComplex */

  /**
   * Get real part
   *
   * @return real part
   */
  CUDA_DEVICE CUDA_HOST
  T real () const
  {
    return re;
  } /* CComplex::real */

  /**
   * Get imaginary part
   *
   * @return imaginary part
   */
  CUDA_DEVICE CUDA_HOST
  T imag () const
  {
    return im;
  } /* CComplex::imag */

  /**
   * Get absolute value of complex
   *
   * @return absolute value of complex
   */
  CUDA_DEVICE CUDA_HOST
  T abs () const
  {
    return sqrt (SQR (re) + SQR (im));
  } /* CComplex::abs */

  /**
   * Get norm value of complex
   *
   * @return norm value of complex
   */
  CUDA_DEVICE CUDA_HOST
  T norm () const
  {
    return SQR (re) + SQR (im);
  } /* CComplex::norm */

  /**
   * Get exponent of complex
   *
   * @return exponent of complex
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> exp () const
  {
    return CComplex<T> (cos (im), sin (im)) * ::exp (re);
  } /* CComplex::exp */

  /**
   * Operator ==
   *
   * @return result of comparison
   */
  CUDA_DEVICE CUDA_HOST
  bool operator== (const CComplex<T> & x) const /**< argument of comparison */
  {
    return re == x.re && im == x.im;
  } /* CComplex::operator== */

  /**
   * Operator !=
   *
   * @return result of comparison
   */
  CUDA_DEVICE CUDA_HOST
  bool operator!= (const CComplex<T> & x) const /**< argument of comparison */
  {
    return re != x.re || im != x.im;
  } /* CComplex::operator!= */

  /**
   * Operator *
   *
   * @return result of multiplication
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator * (const T & x) const /**< multiplier */
  {
    return CComplex (re * x, im * x);
  } /* CComplex::operator* */

  /**
   * Operator *
   *
   * @return result of multiplication
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator * (const CComplex<T> & x) const /**< multiplier */
  {
    return CComplex (re * x.re - im * x.im, im * x.re + re * x.im);
  } /* CComplex::operator* */

  /**
   * Operator /
   *
   * @return result of division
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator / (const T & x) const /**< divider */
  {
    return CComplex (re / x, im / x);
  } /* CComplex::operator/ */

  /**
   * Operator /
   *
   * @return result of division
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator / (const CComplex<T> & x) const /**< divider */
  {
    T d = SQR (x.re) + SQR (x.im);
    return CComplex ((re * x.re + im * x.im) / d, (x.re * im - re * x.im) / d);
  } /* CComplex::operator/ */

  /**
   * Operator +
   *
   * @return result of addition
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator + (const CComplex<T> & x) const /**< operand */
  {
    return CComplex (re + x.re, im + x.im);
  } /* CComplex::operator+ */

  /**
   * Operator -
   *
   * @return result of substraction
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator - (const CComplex<T> & x) const /**< operand */
  {
    return CComplex (re - x.re, im - x.im);
  } /* CComplex::operator- */

  /**
   * Operator +=
   *
   * @return result of addition
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> & operator += (const CComplex<T> & x) /**< operand */
  {
    re += x.re;
    im += x.im;
    return *this;
  } /* CComplex::operator+= */

  /**
   * Operator -=
   *
   * @return result of substraction
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> & operator -= (const CComplex<T> & x) /**< operand */
  {
    re -= x.re;
    im -= x.im;
    return *this;
  } /* CComplex::operator-= */

  /**
   * Operator *=
   *
   * @return result of multiplication
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> & operator *= (const CComplex<T> & x) /**< operand */
  {
    *this = *this * x;
    return *this;
  } /* CComplex::operator+= */

  /**
   * Unary minus operator
   *
   * @return result of unary minus
   */
  CUDA_DEVICE CUDA_HOST
  CComplex<T> operator - () const
  {
    return CComplex (-re, -im);
  } /* CComplex::operator- */
}; /* CComplex */

#endif /* !COMPLEX_H */

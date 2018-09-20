#ifndef FIELD_POINT_H
#define FIELD_POINT_H

#include "FieldValue.h"

/**
 * Class defining all values in time at the specific grid point.
 */
class FieldPointValue
{
  // Current value in time.
  FieldValue currentValue;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // Previous value in time.
  FieldValue previousValue;

#ifdef TWO_TIME_STEPS
  // Previous for previous value in time.
  FieldValue previousPreviousValue;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

public:

  /**
   * Constructor
   *
   * Cuda note: constructor should be empty!
   */
  CUDA_DEVICE CUDA_HOST
  FieldPointValue (
    const FieldValue& curVal = getFieldValueRealOnly (0.0)
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    , const FieldValue& prevVal = getFieldValueRealOnly (0.0)
#ifdef TWO_TIME_STEPS
    , const FieldValue& prevPrevVal = getFieldValueRealOnly (0.0)
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  )
    : currentValue (curVal)
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    , previousValue (prevVal)
#if defined (TWO_TIME_STEPS)
    , previousPreviousValue (prevPrevVal)
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  {
  }

  /**
   * Destructor
   *
   * Cuda note: destructor should be empty!
   */
  CUDA_DEVICE CUDA_HOST
  ~FieldPointValue () {}

  // Getter and setter for current value.
  CUDA_DEVICE CUDA_HOST const FieldValue& getCurValue () const
  {
    return currentValue;
  }
  CUDA_DEVICE CUDA_HOST void setCurValue (const FieldValue& val)
  {
    currentValue = val;
  }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // Getter and setter for previous value.
  CUDA_DEVICE CUDA_HOST const FieldValue& getPrevValue () const
  {
    return previousValue;
  }
  CUDA_DEVICE CUDA_HOST void setPrevValue (const FieldValue& val)
  {
    previousValue = val;
  }

#ifdef TWO_TIME_STEPS
  // Getter and setter for previous for previous value.
  CUDA_DEVICE CUDA_HOST const FieldValue& getPrevPrevValue () const
  {
    return previousPreviousValue;
  }
  CUDA_DEVICE CUDA_HOST void setPrevPrevValue (const FieldValue& val)
  {
    previousPreviousValue = val;
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  // Set all values zero.
  CUDA_DEVICE CUDA_HOST void setZero ()
  {
    currentValue = 0;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    previousValue = 0;

#ifdef TWO_TIME_STEPS
    previousPreviousValue = 0;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  }

  // Replace previous value with current and so on.
  CUDA_DEVICE CUDA_HOST void shiftInTime ()
  {
#ifdef TWO_TIME_STEPS
    previousPreviousValue = previousValue;
#endif /* TWO_TIME_STEPS */

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    previousValue = currentValue;
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  }


  CUDA_DEVICE CUDA_HOST
  bool operator == (FieldPointValue &rhs)
  {
    bool is_same = true;

    is_same &= currentValue == rhs.currentValue;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    is_same &= previousValue == rhs.previousValue;

#ifdef TWO_TIME_STEPS
    is_same &= previousPreviousValue == rhs.previousPreviousValue;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

    return is_same;
  }
};

#endif /* FIELD_POINT_H */

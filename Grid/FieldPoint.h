#ifndef FIELD_POINT_H
#define FIELD_POINT_H

#include <mpi.h>

// Small values define all field values as float.
// Full values define all field values as double.
// Small values decrease memory usage, full values increase accuracy.
#ifdef FLOAT_VALUES
typedef float FieldValue;
#endif /* FLOAT_VALUES */
#ifdef DOUBLE_VALUES
typedef double FieldValue;
#endif /* DOUBLE_VALUES */
#ifdef LONG_DOUBLE_VALUES
typedef long double FieldValue;
#endif /* LONG_DOUBLE_VALUES */


// FieldPointValue defines all values in time at the grid point.
class FieldPointValue
{
  // Current value in time.
  FieldValue currentValue;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // Previous value in time.
  FieldValue previousValue;

#if defined (TWO_TIME_STEPS)
  // Previous for previous value in time.
  FieldValue previousPreviousValue;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

public:

  // Constructor for all cases.
  FieldPointValue (
    const FieldValue& curVal = 0
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    , const FieldValue& prevVal = 0
#if defined (TWO_TIME_STEPS)
    , const FieldValue& prevPrevVal = 0
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  );

  ~FieldPointValue ();

  // Getter and setter for current value.
  const FieldValue& getCurValue () const;
  void setCurValue (const FieldValue& val);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // Getter and setter for previous value.
  const FieldValue& getPrevValue () const;
  void setPrevValue (const FieldValue& val);

#if defined (TWO_TIME_STEPS)
  // Getter and setter for previous for previous value.
  const FieldValue& getPrevPrevValue () const;
  void setPrevPrevValue (const FieldValue& val);
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  // Set all values zero.
  void setZero ();

  // Replace previous value with current and so on.
  void shiftInTime ();
};


#endif /* FIELD_POINT_H */

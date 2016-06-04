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

  // Constructor for all cases.
  FieldPointValue (
    const FieldValue& curVal = 0
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    , const FieldValue& prevVal = 0
#ifdef TWO_TIME_STEPS
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

#ifdef TWO_TIME_STEPS
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

#ifndef FIELD_POINT_H
#define FIELD_POINT_H

// Small values define all field values as float.
// Full values define all field values as double.
// Small values decrease memory usage, full values increase accuracy.
#if defined (SMALL_VALUES)
typedef float FieldValue;
#else
#if defined (FULL_VALUES)
typedef double FieldValue;
#endif
#endif


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
#endif
#endif

public:

  // Constructor for all cases.
  FieldPointValue (
    const FieldValue& curVal = 0
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    , const FieldValue& prevVal = 0
#if defined (TWO_TIME_STEPS)
    , const FieldValue& prevPrevVal = 0
#endif
#endif
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
#endif
#endif

  // Set all values zero.
  void setZero ();

  // Replace previous value with current and so on.
  void shiftInTime ();
};


#endif /* FIELD_POINT_H */

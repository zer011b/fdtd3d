#ifndef FIELD_POINT_H
#define FIELD_POINT_H

/**
 * Small values define all field values as float.
 * Full values define all field values as double.
 * Small values decrease memory usage, full values increase accuracy.
 */
#if defined (SMALL_VALUES)
typedef float FieldValue;
#else
#if defined (FULL_VALUES)
typedef double FieldValue;
#endif
#endif


/**
 * FieldPointValue defines all values in time at the grid point.
 */
class FieldPointValue
{
  FieldValue currentValue;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue previousValue;

#if defined (TWO_TIME_STEPS)
  FieldValue previousPreviousValue;
#endif
#endif

public:

  // Constructor for all cases
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

  const FieldValue& getCurValue () const;
  void setCurValue (const FieldValue& val);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  const FieldValue& getPrevValue () const;
  void setPrevValue (const FieldValue& val);

#if defined (TWO_TIME_STEPS)
  const FieldValue& getPrevPrevValue () const;
  void setPrevPrevValue (const FieldValue& val);
#endif
#endif

  void setZero ();
};


#endif /* FIELD_POINT_H */

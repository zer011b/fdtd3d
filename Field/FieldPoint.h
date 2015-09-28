#ifndef FIELD_POINT_VALUE_H
#define FIELD_POINT_VALUE_H

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
private:
  FieldValue currentValue;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEP)
  FieldValue previousValue;

#if defined (TWO_TIME_STEP)
  FieldValue previousPreviousValue;
#endif
#endif

public:
  FieldPointValue ();
  ~FieldPointValue ();

  FieldValue getCurrentValue ();
  void setCurrentValue (FieldValue val);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEP)
  FieldValue getPrevValue ();
  void setPrevValue (FieldValue val);

#if defined (TWO_TIME_STEP)
  FieldValue getPrevPrevValue ();
  void setPrevPrevValue (FieldValue val);
#endif
#endif
};

#endif /* FIELD_POINT_VALUE_H */
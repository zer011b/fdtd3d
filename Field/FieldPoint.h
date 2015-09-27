#ifndef FIELD_POINT_VALUE_H
#define FIELD_POINT_VALUE_H

/**
 * Small values define all field values as float.
 * Full values define all field values as double.
 * Small values decrease memory usage, full values increase accuracy.
 */
#ifdef SMALL_VALUES
typedef float FieldValue;
#else
#ifdef FULL_VALUES
typedef double FieldValue;
#endif
#endif

/**
 * FieldPointValue defines all values in time at the grid point.
 */
class FieldPointValue
{
private:
  FieldValue* values; 
public:
  FieldPointValue ();
  ~FieldPointValue ();
};

#endif /* FIELD_POINT_VALUE_H */
#ifndef FIELD_VALUES_H
#define FIELD_VALUES_H

/**
 * Small values define all field values as float.
 * Full values define all field values as double.
 * Small values decrease memory usage, full values increase accuracy.
 */
#ifdef FLOAT_VALUES
typedef float FieldValue;
#endif /* FLOAT_VALUES */
#ifdef DOUBLE_VALUES
typedef double FieldValue;
#endif /* DOUBLE_VALUES */
#ifdef LONG_DOUBLE_VALUES
typedef long double FieldValue;
#endif /* LONG_DOUBLE_VALUES */

#endif /* FIELD_VALUES_H */

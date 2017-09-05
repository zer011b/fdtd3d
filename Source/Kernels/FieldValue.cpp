#include "FieldValue.h"

FieldValue getFieldValueRealOnly (FPValue real)
{
#ifdef COMPLEX_FIELD_VALUES
  return FieldValue (real, 0.0);
#else /* COMPLEX_FIELD_VALUES*/
  return FieldValue (real);
#endif /* !COMPLEX_FIELD_VALUES */
}

FPValue getRealOnlyFromFieldValue (const FieldValue &val)
{
#ifdef COMPLEX_FIELD_VALUES
  return val.real ();
#else /* COMPLEX_FIELD_VALUES*/
  return val;
#endif /* !COMPLEX_FIELD_VALUES */
}

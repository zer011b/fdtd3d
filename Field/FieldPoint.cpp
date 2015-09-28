#include "FieldPoint.h"

FieldPointValue::FieldPointValue ()
{
}

FieldPointValue::~FieldPointValue ()
{
}

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEP)
FieldValue FieldPointValue::getPrevValue ()
{
  return previousValue;
}

void FieldPointValue::setPrevValue (FieldValue val)
{
  previousValue = val;
}

#if defined (TWO_TIME_STEP)
FieldValue FieldPointValue::getPrevPrevValue ()
{
  return previousPreviousValue;
}

void FieldPointValue::setPrevPrevValue (FieldValue val)
{
  previousPreviousValue = val;
}
#endif
#endif

#include "FieldPoint.h"

// ================================ FieldPointValue ================================
/**
 * Initialize all values. 
 */
FieldPointValue::FieldPointValue (
  FieldValue curVal
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  , FieldValue prevVal
#if defined (TWO_TIME_STEPS)
  , FieldValue prevPrevVal
#endif
#endif
  ) :
  currentValue (curVal)
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  , previousValue (prevVal)
#if defined (TWO_TIME_STEPS)
  , previousPreviousValue (prevPrevVal)
#endif
#endif
{
}

/**
 * Destructor. Empty.
 */
FieldPointValue::~FieldPointValue ()
{
}

/**
 * Getter/Setter for current value.
 */
FieldValue
FieldPointValue::getCurrentValue ()
{
  return currentValue;
}
void
FieldPointValue::setCurrentValue (FieldValue val)
{
  currentValue = val;
}

/**
 * Getter/Setter for previous value.
 */
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
FieldValue
FieldPointValue::getPrevValue ()
{
  return previousValue;
}
void
FieldPointValue::setPrevValue (FieldValue val)
{
  previousValue = val;
}

/**
 * Getter/Setter for the second previous value.
 */
#if defined (TWO_TIME_STEPS)
FieldValue
FieldPointValue::getPrevPrevValue ()
{
  return previousPreviousValue;
}
void
FieldPointValue::setPrevPrevValue (FieldValue val)
{
  previousPreviousValue = val;
}
#endif
#endif

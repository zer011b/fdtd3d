#include "FieldPoint.h"

/**
 * Initialize all values with zeros. 
 */
FieldPointValue::FieldPointValue ()
{
  currentValue = 0;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = 0;

#if defined (TWO_TIME_STEPS)
  previousPreviousValue = 0;
#endif
#endif
}

/**
 * Initialize current value and all others with zero. 
 */
FieldPointValue::FieldPointValue (FieldValue curVal) :
  currentValue (curVal)
{
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = 0;

#if defined (TWO_TIME_STEPS)
  previousPreviousValue = 0;
#endif
#endif
}

/**
 * Initialize current, previous values and all others with zero. 
 */
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
FieldPointValue::FieldPointValue (FieldValue curVal, FieldValue prevVal) :
  currentValue (curVal), previousValue (prevVal)
{
#if defined (TWO_TIME_STEPS)
  previousPreviousValue = 0;
#endif
}

/**
 * Initialize current and two previous values and all others with zero. 
 */
#if defined (TWO_TIME_STEPS)
FieldPointValue::FieldPointValue (FieldValue curVal, FieldValue prevVal, FieldValue prevPrevVal) :
  currentValue (curVal), previousValue (prevVal), previousPreviousValue (prevPrevVal)
{
}
#endif
#endif

/**
 * Destructor. Empty.
 */
FieldPointValue::~FieldPointValue ()
{
}

/**
 * Getter/Setter for current value.
 */
FieldValue FieldPointValue::getCurrentValue ()
{
  return currentValue;
}
void FieldPointValue::setCurrentValue (FieldValue val)
{
  currentValue = val;
}

/**
 * Getter/Setter for previous value.
 */
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
FieldValue FieldPointValue::getPrevValue ()
{
  return previousValue;
}
void FieldPointValue::setPrevValue (FieldValue val)
{
  previousValue = val;
}

/**
 * Getter/Setter for the second previous value.
 */
#if defined (TWO_TIME_STEPS)
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

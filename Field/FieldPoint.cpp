#include "FieldPoint.h"

// ================================ FieldPointValue ================================
/**
 * Initialize all values.
 */
FieldPointValue::FieldPointValue (
  const FieldValue& curVal
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  , const FieldValue& prevVal
#if defined (TWO_TIME_STEPS)
  , const FieldValue& prevPrevVal
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
const FieldValue&
FieldPointValue::getCurValue () const
{
  return currentValue;
}
void
FieldPointValue::setCurValue (const FieldValue& val)
{
  currentValue = val;
}

/**
 * Getter/Setter for previous value.
 */
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
const FieldValue&
FieldPointValue::getPrevValue () const
{
  return previousValue;
}
void
FieldPointValue::setPrevValue (const FieldValue& val)
{
  previousValue = val;
}

/**
 * Getter/Setter for the second previous value.
 */
#if defined (TWO_TIME_STEPS)
const FieldValue&
FieldPointValue::getPrevPrevValue () const
{
  return previousPreviousValue;
}
void
FieldPointValue::setPrevPrevValue (const FieldValue& val)
{
  previousPreviousValue = val;
}
#endif
#endif

void
FieldPointValue::setZero ()
{
  currentValue = 0;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = 0;

#if defined (TWO_TIME_STEPS)
  previousPreviousValue = 0;
#endif
#endif
}

void
FieldPointValue::shiftInTime ()
{
#if defined (TWO_TIME_STEPS)
  previousPreviousValue = previousValue;
#endif

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = currentValue;
#endif
}

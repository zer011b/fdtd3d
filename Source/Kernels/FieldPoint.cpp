#include "FieldPoint.h"

// ================================ FieldPointValue ================================
FieldPointValue::FieldPointValue (
  const FieldValue& curVal
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  , const FieldValue& prevVal
#if defined (TWO_TIME_STEPS)
  , const FieldValue& prevPrevVal
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  ) :
  currentValue (curVal)
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  , previousValue (prevVal)
#if defined (TWO_TIME_STEPS)
  , previousPreviousValue (prevPrevVal)
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
{
}

FieldPointValue::~FieldPointValue ()
{
}


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


#ifdef TWO_TIME_STEPS
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
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */


void
FieldPointValue::setZero ()
{
  currentValue = 0;

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = 0;

#ifdef TWO_TIME_STEPS
  previousPreviousValue = 0;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}


void
FieldPointValue::shiftInTime ()
{
#ifdef TWO_TIME_STEPS
  previousPreviousValue = previousValue;
#endif /* TWO_TIME_STEPS */

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  previousValue = currentValue;
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

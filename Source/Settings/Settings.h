#ifndef SETTINGS_H
#define SETTINGS_H

#include "FieldValue.h"
#include "Assert.h"
#include "PhysicsConst.h"

#define SOLVER_VERSION "0.2.2"

/**
 * Log levels
 */
ENUM_CLASS (LogLevelType, uint8_t,
  LOG_LEVEL_0 = 0,
  LOG_LEVEL_NONE = LOG_LEVEL_0,
  LOG_LEVEL_1,
  LOG_LEVEL_STAGES = LOG_LEVEL_1,
  LOG_LEVEL_2,
  LOG_LEVEL_STAGES_AND_DUMP = LOG_LEVEL_2,
  LOG_LEVEL_3,
  LOG_LEVEL_FULL = LOG_LEVEL_3
);

ENUM_CLASS (ArgType, uint8_t,
  TYPE_NONE,
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_STRING,
  TYPE_LOG_LEVEL
);

/**
 * Settings for solver
 */
class Settings
{
private:

  /**
   * Number of dimensions
   */
  int dimension;

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  fieldType fieldName;
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_OPTION(cmdArg, hasArg, argType, description)
#include "Settings.inc"

public:

  /**
   * Default constructor
   */
  Settings ()
    : dimension (0)
#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    , fieldName (defaultVal)
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_OPTION(cmdArg, hasArg, argType, description)
#include "Settings.inc"
  {
  } /* Settings */

  /**
   * Destructor
   */
  ~Settings ()
  {
  } /* ~Settings */

  void setFromCmd (int, char **);

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  fieldType getterName () \
  { \
    return fieldName; \
  }
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_OPTION(cmdArg, hasArg, argType, description)
#include "Settings.inc"

  /**
   * Get number of dimensions
   *
   * @return number of dimensions
   */
  int getDimension () const
  {
    return dimension;
  } /* Settings::getDimension */
}; /* Settings */

#endif /* !SETTINGS_H */

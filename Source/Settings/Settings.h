#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>

#include "Assert.h"
#include "FieldValue.h"
#include "PhysicsConst.h"

/**
 * Log levels
 */
enum LogLevelType
{
  LOG_LEVEL_0 = 0,
  LOG_LEVEL_NONE = LOG_LEVEL_0,
  LOG_LEVEL_1,
  LOG_LEVEL_STAGES = LOG_LEVEL_1,
  LOG_LEVEL_2,
  LOG_LEVEL_STAGES_AND_DUMP = LOG_LEVEL_2,
  LOG_LEVEL_3,
  LOG_LEVEL_FULL = LOG_LEVEL_3
};

ENUM_CLASS (SchemeType, uint8_t,
  NONE,
  Dim1_ExHy,
  Dim1_ExHz,
  Dim1_EyHx,
  Dim1_EyHz,
  Dim1_EzHx,
  Dim1_EzHy,
  Dim2_TEx,
  Dim2_TEy,
  Dim2_TEz,
  Dim2_TMx,
  Dim2_TMy,
  Dim2_TMz,
  Dim3
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

  /**
   * Type of calculation scheme
   */
  SchemeType schemeType;

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
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description)
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description)
#include "Settings.inc.h"

private:

  int parseArg (int &, int, char **, bool);
  int setFromCmd (int, char **, bool);
  int loadCmdFromFile (std::string);
  int saveCmdToFile (int, char **, std::string);

public:

  /**
   * Default constructor
   */
  Settings ()
    : dimension (0)
    , schemeType (SchemeType::NONE)
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
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description)
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description)
#include "Settings.inc.h"
  {
  } /* Settings */

  /**
   * Destructor
   */
  ~Settings ()
  {
  } /* ~Settings */

  void SetupFromCmd (int, char **);

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
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description)
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description)
#include "Settings.inc.h"

  /**
   * Get number of dimensions
   *
   * @return number of dimensions
   */
  int getDimension () const
  {
    return dimension;
  } /* Settings::getDimension */

  /**
   * Get scheme type
   *
   * @return scheme type
   */
  SchemeType getSchemeType () const
  {
    return schemeType;
  } /* Settings::getSchemeType */
}; /* Settings */

extern Settings solverSettings;

#endif /* !SETTINGS_H */

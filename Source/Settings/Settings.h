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

typedef uint8_t SchemeType_t;

ENUM_CLASS (SchemeType, SchemeType_t,
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

  Settings *d_cudaSolverSettings;

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

  CUDA_HOST int parseArg (int &, int, char **, bool);
  CUDA_HOST int setFromCmd (int, char **, bool);
  CUDA_HOST int loadCmdFromFile (std::string);
  CUDA_HOST int saveCmdToFile (int, char **, std::string);

#ifdef CUDA_ENABLED
  CUDA_HOST void prepareDeviceSettings ();
  CUDA_HOST void freeDeviceSettings ();
#endif /* CUDA_ENABLED */

public:

  /**
   * Default constructor
   */
  CUDA_HOST
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

  CUDA_HOST ~Settings () {}

  CUDA_HOST void Initialize ();
  CUDA_HOST void Uninitialize ();

  CUDA_HOST
  void SetupFromCmd (int, char **);

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  CUDA_DEVICE CUDA_HOST fieldType getterName () \
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
   * Get pointer to settings object, which is allocated on GPU
   *
   * @return pointer to settings object, which is allocated on GPU
   */
  Settings *getCudaSettings () const
  {
    return d_cudaSolverSettings;
  } /* Settings::getCudaSettings */

  /**
   * Get number of dimensions
   *
   * @return number of dimensions
   */
  CUDA_DEVICE CUDA_HOST
  int getDimension () const
  {
    return dimension;
  } /* Settings::getDimension */

  /**
   * Get scheme type
   *
   * @return scheme type
   */
  CUDA_DEVICE CUDA_HOST
  SchemeType getSchemeType () const
  {
    return schemeType;
  } /* Settings::getSchemeType */
}; /* Settings */

#ifdef CUDA_ENABLED
#if ! defined (SETTINGS_CU) && defined (CUDA_SOURCES)
extern __constant__ Settings *cudaSolverSettings;
#endif
#endif /* CUDA_ENABLED */

extern Settings solverSettings;

#endif /* !SETTINGS_H */

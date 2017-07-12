#include "Settings.h"

#include <string.h>

#ifdef CXX11_ENABLED
#else
#include <stdlib.h>
#endif

Settings solverSettings;

/**
 * Set settings from command line arguments
 */
void
Settings::setFromCmd (int argc, /**< number of arguments */
                      char **argv) /**< arguments */
{
  if (argc == 0)
  {
    exit (EXIT_OK);
  }

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp (argv[i], "--help") == 0)
    {
      printf ("fdtd3d is an open source 1D [NYI], 2D, 3D FDTD electromagnetics solver with MPI, OpenMP [NYI] and CUDA [WIP] support.\n");
      printf ("Usage: fdtd3d [options]\n\n");
      printf ("Options:\n");

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
      printf ("  %s\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
      SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
      SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
      SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
      SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description)
#define SETTINGS_ELEM_OPTION(cmdArg, hasArg, argType, description) \
      printf ("  %s\n\t%s\n", cmdArg, description);
#include "Settings.inc"

      exit (EXIT_OK);
    }
    else if (strcmp (argv[i], "--version") == 0)
    {
      printf ("Version: %s\n", SOLVER_VERSION);

      exit (EXIT_OK);
    }

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    else if (strcmp (argv[i], cmdArg) == 0) \
    { \
      fieldName = true; \
    }
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    else if (strcmp (argv[i], cmdArg) == 0) \
    { \
      ++i; \
      fieldName = STOI (argv[i]); \
    }
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    else if (strcmp (argv[i], cmdArg) == 0) \
    { \
      ++i; \
      fieldName = STOF (argv[i]); \
    }
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    else if (strcmp (argv[i], cmdArg) == 0) \
    { \
      ++i; \
      fieldName = std::string (argv[i]); \
    }
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    else if (strcmp (argv[i], cmdArg) == 0) \
    { \
      ++i; \
      switch (STOI (argv[i])) \
      { \
        case 0: \
        { \
          fieldName = LogLevelType::LOG_LEVEL_0; \
          break; \
        } \
        case 1: \
        { \
          logLevel = LogLevelType::LOG_LEVEL_1; \
          break; \
        } \
        case 2: \
        { \
          logLevel = LogLevelType::LOG_LEVEL_2; \
          break; \
        } \
        case 3: \
        { \
          logLevel = LogLevelType::LOG_LEVEL_3; \
          break; \
        } \
        default: \
        { \
          UNREACHABLE; \
        } \
      } \
    }
#define SETTINGS_ELEM_OPTION(cmdArg, hasArg, argType, description)
#include "Settings.inc"

    else if (strcmp (argv[i], "--same-size") == 0)
    {
      sizeZ = sizeY = sizeX;
    }
    else if (strcmp (argv[i], "--same-size-pml") == 0)
    {
      pmlSizeZ = pmlSizeY = pmlSizeX;
    }
    else if (strcmp (argv[i], "--same-size-tfsf") == 0)
    {
      tfsfSizeZ = tfsfSizeY = tfsfSizeX;
    }
    else if (strcmp (argv[i], "--same-size-ntff") == 0)
    {
      ntffSizeZ = tfsfSizeY = tfsfSizeX;
    }
    else if (strcmp (argv[i], "--2d") == 0)
    {
      dimension = 2;
    }
    else if (strcmp (argv[i], "--3d") == 0)
    {
      dimension = 3;
    }
    else
    {
      printf ("Unknown option [%s]\n", argv[i]);
      exit (EXIT_UNKNOWN_OPTION);
    }
  }
} /* Settings::setFromCmd */

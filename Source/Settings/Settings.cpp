#include "Settings.h"

#include <string.h>
#include <fstream>
#include <cstring>

#ifdef CXX11_ENABLED
#else
#include <stdlib.h>
#endif

Settings solverSettings;

/**
 * Parse single command line argument
 *
 * @return exit code
 */
int
Settings::parseArg (int &index, /**< out: current argument index */
                    int argc, /**< total number of indexes */
                    char **argv) /**< vector of cmd args */
{
  ASSERT (index >= 0 && index < argc);

  if (strcmp (argv[index], "--help") == 0)
  {
    printf ("fdtd3d is an open source 1D [NYI], 2D, 3D FDTD electromagnetics solver with MPI, OpenMP [NYI] and CUDA [WIP] support.\n");
    printf ("Usage: fdtd3d [options]\n\n");
    printf ("Options:\n");

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <int>\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <float>\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <string>\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <int>\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description) \
    printf ("  %s\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description) \
    printf ("  %s <string>\n\t%s\n", cmdArg, description);
#include "Settings.inc"

    return EXIT_BREAK_ARG_PARSING;
  }
  else if (strcmp (argv[index], "--version") == 0)
  {
    printf ("Version: %s\n", SOLVER_VERSION);

    return EXIT_BREAK_ARG_PARSING;
  }

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    fieldName = true; \
  }
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    fieldName = STOI (argv[index]); \
  }
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    fieldName = STOF (argv[index]); \
  }
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    fieldName = std::string (argv[index]); \
  }
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    switch (STOI (argv[index])) \
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
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description)
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description)
#include "Settings.inc"

  else if (strcmp (argv[index], "--same-size") == 0)
  {
    sizeZ = sizeY = sizeX;
  }
  else if (strcmp (argv[index], "--same-size-pml") == 0)
  {
    pmlSizeZ = pmlSizeY = pmlSizeX;
  }
  else if (strcmp (argv[index], "--same-size-tfsf") == 0)
  {
    tfsfSizeZ = tfsfSizeY = tfsfSizeX;
  }
  else if (strcmp (argv[index], "--same-size-ntff") == 0)
  {
    ntffSizeZ = tfsfSizeY = tfsfSizeX;
  }
  else if (strcmp (argv[index], "--2d") == 0)
  {
    dimension = 2;
  }
  else if (strcmp (argv[index], "--3d") == 0)
  {
    dimension = 3;
  }
  else if (strcmp (argv[index], "--cmd-from-file") == 0)
  {
    if (argc != 3)
    {
      printf ("Command line files are allowed only without other options.\n");
      return EXIT_ERROR;
    }

    ++index;
    ASSERT (index >= 0 && index < argc);
    int status = loadFromFile (argv[index]);

    if (status == EXIT_ERROR)
    {
      printf ("ERROR: Incorrect command line file.\n");
    }

    return status;
  }
  else
  {
    printf ("Unknown option [%s]\n", argv[index]);
    return EXIT_UNKNOWN_OPTION;
  }

  return EXIT_OK;
} /* Settings::parseArg */

/**
 * Set settings from command line arguments
 *
 * @return exit code
 */
int
Settings::setFromCmd (int start_index, /**< index to start arguments parsing */
                      int argc, /**< number of arguments */
                      char **argv) /**< arguments */
{
  for (int i = start_index; i < argc; ++i)
  {
    int status = parseArg (i, argc, argv);

    switch (status)
    {
      case EXIT_OK:
      {
        break;
      }
      case EXIT_BREAK_ARG_PARSING:
      case EXIT_ERROR:
      case EXIT_UNKNOWN_OPTION:
      {
        return status;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  return EXIT_OK;
} /* Settings::setFromCmd */

/**
 * Load command line arguments from file
 *
 * @return exit code
 */
int
Settings::loadFromFile (std::string fileName) /**< name of file to load from */
{
  printf ("Loading command line from file %s\n", fileName.c_str ());

  std::string cmd;

  std::ifstream infile(fileName);

  int argc = 0;
  while (infile >> cmd)
  {
    ++argc;
  }

  infile.close ();
  infile.open (fileName);

  char **argv = new char *[argc];

  int index = 0;
  while (infile >> cmd)
  {
    argv[index] = new char[cmd.length ()];
    strcpy (argv[index], cmd.c_str ());
    ++index;
  }

  int status = setFromCmd (0, argc, argv);

  for (int i = 0; i < argc; ++i)
  {
    delete[] argv[i];
  }
  delete[] argv;

  infile.close ();

  return status;
} /* Settings::loadFromFile */

int
Settings::saveToFile (std::string fileName) /**< name of file to save to */
{

}

/**
 * Set settings from command line arguments
 */
void
Settings::SetupFromCmd (int argc, /**< number of arguments */
                        char **argv) /**< arguments */
{
  int status = setFromCmd (1, argc, argv);

  switch (status)
  {
    case EXIT_OK:
    {
      break;
    }
    case EXIT_BREAK_ARG_PARSING:
    {
      exit (EXIT_OK);
    }
    case EXIT_ERROR:
    case EXIT_UNKNOWN_OPTION:
    {
      exit (status);
    }
  }
} /* Settings::SetupFromCmd */

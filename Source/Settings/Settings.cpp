/*
 * Copyright (C) 2017 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "Settings.h"

#include <alloca.h>
#include <string.h>
#include <fstream>
#include <cstring>
#include <sstream>

#ifndef CXX11_ENABLED
#include <stdlib.h>
#endif

Settings solverSettings;

/**
 * Perform initialization of settings
 */
CUDA_HOST
void
Settings::Initialize ()
{
#ifdef CUDA_ENABLED
  if (getDoUseCuda ())
  {
    prepareDeviceSettings ();
  }
#endif /* CUDA_ENABLED */

  isInitialized = true;
} /* Settings::Initialize */

/**
 * Perform uninitialization of settings
 */
CUDA_HOST
void
Settings::Uninitialize ()
{
#ifdef CUDA_ENABLED
  if (getDoUseCuda ())
  {
    freeDeviceSettings ();
  }
#endif /* CUDA_ENABLED */

  ASSERT (isInitialized);
} /* Settings::Uninitialize */

/**
 * Parse coordinate in x:int,y:int,z:int format
 */
CUDA_HOST
void
Settings::parseCoordinate (const char *str, /**< string to parse */
                           int &xval, /**< out: x value */
                           int &yval, /**< out: y value */
                           int &zval) /**< out: z value */
{
  const char *coordStr = str;
  const int tmpBufSize = 16;
  char * tmpCoordStr = (char *) alloca (tmpBufSize);
  int i = 0;
  while (coordStr[i] != '\0')
  {
    bool isEnd = false;

    while (coordStr[i] != ',' && coordStr[i] != '\0')
    {
      ++i;
    }

    if (coordStr[i] == '\0')
    {
      isEnd = true;
    }

    ASSERT (i < tmpBufSize);
    strncpy (tmpCoordStr, coordStr, i);
    tmpCoordStr[i] = '\0';

    ASSERT (tmpCoordStr[1] == ':');
    int val = STOI (tmpCoordStr+2);

    if (tmpCoordStr[0] == 'x' || tmpCoordStr[0] == 'X')
    {
      xval = val;
    }
    else if (tmpCoordStr[0] == 'y' || tmpCoordStr[0] == 'Y')
    {
      yval = val;
    }
    else if (tmpCoordStr[0] == 'z' || tmpCoordStr[0] == 'Z')
    {
      zval = val;
    }

    if (!isEnd)
    {
      ++i;
      coordStr += i;
      i = 0;
    }
  }
} /* Settings::parseCoordinate */

/**
 * Parse single command line argument
 *
 * @return exit code
 */
CUDA_HOST
int
Settings::parseArg (int &index, /**< out: current argument index */
                    int argc, /**< total number of indexes */
                    const char * const * argv, /**< vector of cmd args */
                    bool isCmd) /**< flag, whether argumens are passed through actual command line */
{
  ASSERT (index >= 0 && index < argc);

  if (strcmp (argv[index], "--help") == 0)
  {
    printf ("fdtd3d is an open source 1D, 2D, 3D FDTD electromagnetics solver with MPI, OpenMP [NYI] and CUDA support.\n");
    printf ("Usage: fdtd3d [options]\n\n");
    printf ("Options:\n");

#define SETTINGS_ELEM_FIELD_TYPE_NONE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_FIELD_TYPE_INT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <int> (default: %d)\n\t%s\n", cmdArg, defaultVal, description);
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <float> (default: %f)\n\t%s\n", cmdArg, defaultVal, description);
#define SETTINGS_ELEM_FIELD_TYPE_STRING(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <string> (default: %s)\n\t%s\n", cmdArg, defaultVal, description);
#define SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s <int> (default: %d)\n\t%s\n", cmdArg, defaultVal, description);
#define SETTINGS_ELEM_FIELD_TYPE_COORDINATE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
    printf ("  %s x:<int>,y:<int>,z:<int> (at least one coordinate should be present, default: %d)\n\t%s\n", cmdArg, defaultVal, description);
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description) \
    printf ("  %s\n\t%s\n", cmdArg, description);
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description) \
    printf ("  %s <string>\n\t%s\n", cmdArg, description);
#include "Settings.inc.h"

    return EXIT_BREAK_ARG_PARSING;
  }
  else if (strcmp (argv[index], "--version") == 0)
  {
    printf ("Version: %.1f\n", SOLVER_VERSION);

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
    fieldName = (fieldType) STOI (argv[index]); \
  }
#define SETTINGS_ELEM_FIELD_TYPE_FLOAT(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    fieldName = (fieldType) STOF (argv[index]); \
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
        fieldName = LOG_LEVEL_0; \
        break; \
      } \
      case 1: \
      { \
        logLevel = LOG_LEVEL_1; \
        break; \
      } \
      case 2: \
      { \
        logLevel = LOG_LEVEL_2; \
        break; \
      } \
      case 3: \
      { \
        logLevel = LOG_LEVEL_3; \
        break; \
      } \
      default: \
      { \
        UNREACHABLE; \
      } \
    } \
  }
#define SETTINGS_ELEM_FIELD_TYPE_COORDINATE(fieldName, getterName, fieldType, defaultVal, cmdArg, description) \
  else if (strcmp (argv[index], cmdArg) == 0) \
  { \
    ++index; \
    ASSERT (index >= 0 && index < argc); \
    int xval = defaultVal; \
    int yval = defaultVal; \
    int zval = defaultVal; \
    parseCoordinate (argv[index], xval, yval, zval); \
    fieldName ## X = (fieldType) xval; \
    fieldName ## Y = (fieldType) yval; \
    fieldName ## Z = (fieldType) zval; \
  }
#define SETTINGS_ELEM_OPTION_TYPE_NONE(cmdArg, description)
#define SETTINGS_ELEM_OPTION_TYPE_STRING(cmdArg, description)
#include "Settings.inc.h"

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
    tfsfSizeLeftZ = tfsfSizeLeftY = tfsfSizeLeftX;
    tfsfSizeRightZ = tfsfSizeRightY = tfsfSizeRightX;
  }
  else if (strcmp (argv[index], "--same-size-ntff") == 0)
  {
    ntffSizeZ = ntffSizeY = ntffSizeX;
  }
  else if (strcmp (argv[index], "--1d-exhy") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_ExHy;
  }
  else if (strcmp (argv[index], "--1d-exhz") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_ExHz;
  }
  else if (strcmp (argv[index], "--1d-eyhx") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_EyHx;
  }
  else if (strcmp (argv[index], "--1d-eyhz") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_EyHz;
  }
  else if (strcmp (argv[index], "--1d-ezhx") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_EzHx;
  }
  else if (strcmp (argv[index], "--1d-ezhy") == 0)
  {
    dimension = 1;
    schemeType = SchemeType::Dim1_EzHy;
  }
  else if (strcmp (argv[index], "--2d-tex") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TEx;
  }
  else if (strcmp (argv[index], "--2d-tey") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TEy;
  }
  else if (strcmp (argv[index], "--2d-tez") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TEz;
  }
  else if (strcmp (argv[index], "--2d-tmx") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TMx;
  }
  else if (strcmp (argv[index], "--2d-tmy") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TMy;
  }
  else if (strcmp (argv[index], "--2d-tmz") == 0)
  {
    dimension = 2;
    schemeType = SchemeType::Dim2_TMz;
  }
  else if (strcmp (argv[index], "--3d") == 0)
  {
    dimension = 3;
    schemeType = SchemeType::Dim3;
  }
  else if (strcmp (argv[index], "--cmd-from-file") == 0)
  {
    if (!isCmd)
    {
      printf ("Command line files are not allowed in other command line files.\n");
      return EXIT_ERROR;
    }

    if (argc != 3)
    {
      printf ("Command line files are allowed only without other options.\n");
      return EXIT_ERROR;
    }

    ++index;
    ASSERT (index >= 0 && index < argc);
    int status = loadCmdFromFile (argv[index]);

    if (status == EXIT_ERROR)
    {
      printf ("ERROR: Incorrect command line file.\n");
    }

    return status;
  }
  else if (strcmp (argv[index], "--save-cmd-to-file") == 0)
  {
    ++index;
    ASSERT (index >= 0 && index < argc);
    int status = saveCmdToFile (argc, argv, argv[index]);
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
CUDA_HOST
int
Settings::setFromCmd (int argc, /**< number of arguments */
                      const char * const * argv, /**< arguments */
                      bool isCmd) /**< flag, whether argumens are passed through actual command line */
{
  if (argc == (isCmd ? 1 : 0))
  {
    printf ("Help: fdtd3d --help\n");
    return EXIT_BREAK_ARG_PARSING;
  }

  for (int i = (isCmd ? 1 : 0); i < argc; ++i)
  {
    int status = parseArg (i, argc, argv, isCmd);

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
CUDA_HOST
int
Settings::loadCmdFromFile (std::string fileName) /**< name of file to load from */
{
  printf ("Loading command line from file %s\n", fileName.c_str ());

  std::ifstream infile;
  infile.open (fileName.c_str ());

  if ((infile.rdstate() & std::ifstream::failbit) != 0)
  {
    printf ("Unable to open file: %s\n", fileName.c_str ());
    return EXIT_ERROR;
  }

  std::string line;
  std::string cmd;
  int argc = 0;
  while (std::getline (infile, line))
  {
    if (line.empty())
    {
      continue;
    }
    else if (line.length () >= 2
             && line[0] == '/'
             && line[1] == '/')
    {
      /*
       * comment
       */
      continue;
    }
    else if (line.length () >= 1
             && line[0] == '#')
    {
      /*
       * comment
       */
      continue;
    }

    std::stringstream ss (line);
    while (ss >> cmd)
    {
      ++argc;
    }
  }

  infile.close ();
  infile.open (fileName.c_str ());

  char **argv = new char *[argc];

  int index = 0;
  while (std::getline (infile, line))
  {
    if (line.empty())
    {
      continue;
    }
    else if (line.length () >= 2
             && line[0] == '/'
             && line[1] == '/')
    {
      /*
       * comment
       */
      continue;
    }
    else if (line.length () >= 1
             && line[0] == '#')
    {
      /*
       * comment
       */
      continue;
    }

    std::stringstream ss (line);
    while (ss >> cmd)
    {
      argv[index] = new char[cmd.length () + 1];
      strcpy (argv[index], cmd.c_str ());
      ++index;
    }
  }

  int status = setFromCmd (argc, argv, false);

  for (int i = 0; i < argc; ++i)
  {
    delete[] argv[i];
  }
  delete[] argv;

  infile.close ();

  return status;
} /* Settings::loadCmdFromFile */

/**
 * Save command line arguments to file
 *
 * @return exit code
 */
CUDA_HOST
int
Settings::saveCmdToFile (int argc, /**< number of arguments */
                         const char * const * argv, /**< arguments */
                         std::string fileName) /**< name of file to save to */
{
  printf ("Saving command line to file %s\n", fileName.c_str ());

  std::ofstream outfile;
  outfile.open (fileName.c_str ());

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp (argv[i], "--save-cmd-to-file") == 0)
    {
      ++i;
    }
    else
    {
      outfile << argv[i] << std::endl;
    }
  }

  outfile.close ();

  return EXIT_OK;
} /* Settings::saveCmdToFile */

/**
 * Set settings from command line arguments
 */
CUDA_HOST
void
Settings::SetupFromCmd (int argc, /**< number of arguments */
                        const char * const *argv) /**< arguments */
{
  int status = setFromCmd (argc, argv, true);

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

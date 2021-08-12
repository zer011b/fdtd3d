/*
 * Copyright (C) 2019 Gleb Balykov
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

#include <iostream>

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>

#include "Settings.h"
#include "GridInterface.h"

int main (int argc, char **argv)
{
  FileType fromType;
  FileType toType;

  grid_coord sizex = 0;
  grid_coord sizey = 0;
  grid_coord sizez = 0;

  grid_coord startx = 0;
  grid_coord starty = 0;
  grid_coord startz = 0;

  grid_coord endx = 0;
  grid_coord endy = 0;
  grid_coord endz = 0;

  FPValue maxPosReManual = 0;
  FPValue maxPosImManual = 0;
  FPValue maxPosModManual = 0;

  FPValue maxNegReManual = 0;
  FPValue maxNegImManual = 0;
  FPValue maxNegModManual = 0;

  bool manualMax = false;

  int dim = 0;

  if (argc == 1)
  {
    printf ("example: ./converter --txt-to-bmp --file 1.txt)\n");
    return 1;
  }

  std::string input;

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp (argv[i], "--file") == 0)
    {
      ++i;
      input = std::string (argv[i]);
    }
    else if (strcmp (argv[i], "--txt-to-bmp") == 0)
    {
      fromType = FILE_TYPE_TXT;
      toType = FILE_TYPE_BMP;
    }
    else if (strcmp (argv[i], "--sizex") == 0)
    {
      ++i;
      sizex = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizey") == 0)
    {
      ++i;
      sizey = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizez") == 0)
    {
      ++i;
      sizez = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--dim") == 0)
    {
      ++i;
      dim = STOI (argv[i]);
      ALWAYS_ASSERT (dim > 0 && dim < 4);
    }
    else if (strcmp (argv[i], "--startx") == 0)
    {
      ++i;
      startx = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--starty") == 0)
    {
      ++i;
      starty = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--startz") == 0)
    {
      ++i;
      startz = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--endx") == 0)
    {
      ++i;
      endx = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--endy") == 0)
    {
      ++i;
      endy = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--endz") == 0)
    {
      ++i;
      endz = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--manualmax") == 0)
    {
      manualMax = true;
    }
    else if (strcmp (argv[i], "--maxposre") == 0)
    {
      ++i;
      maxPosReManual = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--maxnegre") == 0)
    {
      ++i;
      maxNegReManual = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--maxposim") == 0)
    {
      ++i;
      maxPosImManual = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--maxnegim") == 0)
    {
      ++i;
      maxNegImManual = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--maxposmod") == 0)
    {
      ++i;
      maxPosModManual = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--maxnegmod") == 0)
    {
      ++i;
      maxNegModManual = STOF (argv[i]);
    }
    else
    {
      return 1;
    }
  }

  std::string output;
  switch (toType)
  {
    case FILE_TYPE_TXT:
    {
      output = input;
      break;
    }
    case FILE_TYPE_DAT:
    {
      output = input;
      break;
    }
    case FILE_TYPE_BMP:
    {
      output = input;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (dim == 3)
  {
    GridCoordinate3D start = GRID_COORDINATE_3D (startx, starty, startz, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    GridCoordinate3D end = GRID_COORDINATE_3D (endx, endy, endz, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    GridCoordinate3D size = GRID_COORDINATE_3D (sizex, sizey, sizez, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);

    if (fromType == FILE_TYPE_TXT && toType == FILE_TYPE_BMP)
    {
      // TODO: only plain images, stored as txt are currently supported
      ALWAYS_ASSERT (startx + 1 == endx || starty + 1 == endy || startz + 1 == endz);

      BMPHelper BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED, OrthogonalAxis::Z);

      if (startz + 1 == endz)
      {
        std::ifstream file;
        file.open (input.c_str (), std::ios::in);
        ASSERT (file.is_open());

        file >> std::setprecision(std::numeric_limits<double>::digits10);

        BMP imageRe;
        imageRe.SetSize (sizex, sizey);
        imageRe.SetBitDepth (BMPHelper::bitDepth);

        FPValue maxPosRe;
        FPValue maxNegRe;
        std::ofstream fileMaxRe;

#ifdef COMPLEX_FIELD_VALUES
        BMP imageIm;
        imageIm.SetSize (sizex, sizey);
        imageIm.SetBitDepth (BMPHelper::bitDepth);

        FPValue maxPosIm;
        FPValue maxNegIm;
        std::ofstream fileMaxIm;

        BMP imageMod;
        imageMod.SetSize (sizex, sizey);
        imageMod.SetBitDepth (BMPHelper::bitDepth);

        FPValue maxPosMod;
        FPValue maxNegMod;
        std::ofstream fileMaxMod;
#endif /* COMPLEX_FIELD_VALUES */

        for (grid_coord i = startx; i < endx; ++i)
        {
          for (grid_coord j = starty; j < endy; ++j)
          {
            for (grid_coord k = startz; k < endz; ++k)
            {
              std::string line;

              std::getline (file, line);
              ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

              std::string buf;
              std::vector<std::string> tokens;
              std::stringstream ss (line);
              while (ss >> buf)
              {
                tokens.push_back(buf);
              }

              uint32_t word_index = 0;

              ASSERT (i == STOI (tokens[word_index].c_str ()));
              ++word_index;
              ASSERT (j == STOI (tokens[word_index].c_str ()));
              ++word_index;
              ASSERT (k == STOI (tokens[word_index].c_str ()));
              ++word_index;

              FPValue real = STOF (tokens[word_index++].c_str ());
              ASSERT (word_index == 4);
#ifdef COMPLEX_FIELD_VALUES
              FPValue imag = STOF (tokens[word_index++].c_str ());
              ASSERT (word_index == 5);
              FPValue mod = sqrt (SQR (real) + SQR (imag));
#endif

              if (real > maxPosRe
                  || (i == startx && j == starty && k == startz))
              {
                maxPosRe = real;
              }
              if (real < maxNegRe
                  || (i == startx && j == starty && k == startz))
              {
                maxNegRe = real;
              }

#ifdef COMPLEX_FIELD_VALUES
              if (imag > maxPosIm
                  || (i == startx && j == starty && k == startz))
              {
                maxPosIm = imag;
              }
              if (imag < maxNegIm
                  || (i == startx && j == starty && k == startz))
              {
                maxNegIm = imag;
              }

              if (mod > maxPosMod
                  || (i == startx && j == starty && k == startz))
              {
                maxPosMod = mod;
              }
              if (mod < maxNegMod
                  || (i == startx && j == starty && k == startz))
              {
                maxNegMod = mod;
              }
#endif
            }
          }
        }

        file.close ();

        if (manualMax)
        {
          maxPosRe = maxPosReManual;
          maxNegRe = maxNegReManual;

#ifdef COMPLEX_FIELD_VALUES
          maxPosIm = maxPosImManual;
          maxNegIm = maxNegImManual;

          maxPosMod = maxPosModManual;
          maxNegMod = maxNegModManual;
#endif
        }

        file.open (input.c_str (), std::ios::in);
        ASSERT (file.is_open());

        file >> std::setprecision(std::numeric_limits<double>::digits10);

        const FPValue maxRe = maxPosRe - maxNegRe;
#ifdef COMPLEX_FIELD_VALUES
        const FPValue maxIm = maxPosIm - maxNegIm;
        const FPValue maxMod = maxPosMod - maxNegMod;
#endif /* COMPLEX_FIELD_VALUES */

        for (grid_coord i = startx; i < endx; ++i)
        {
          for (grid_coord j = starty; j < endy; ++j)
          {
            for (grid_coord k = startz; k < endz; ++k)
            {
              std::string line;

              std::getline (file, line);
              ASSERT ((file.rdstate() & std::ifstream::failbit) == 0);

              std::string buf;
              std::vector<std::string> tokens;
              std::stringstream ss (line);
              while (ss >> buf)
              {
                tokens.push_back(buf);
              }

              uint32_t word_index = 0;

              ASSERT (i == STOI (tokens[word_index].c_str ()));
              ++word_index;
              ASSERT (j == STOI (tokens[word_index].c_str ()));
              ++word_index;
              ASSERT (k == STOI (tokens[word_index].c_str ()));
              ++word_index;

              FPValue real = STOF (tokens[word_index++].c_str ());
              ASSERT (word_index == 4);
#ifdef COMPLEX_FIELD_VALUES
              FPValue imag = STOF (tokens[word_index++].c_str ());
              ASSERT (word_index == 5);
              FPValue mod = sqrt (SQR (real) + SQR (imag));
#endif

              RGBApixel pixelRe = BMPhelper.getPixelFromValue (real, maxNegRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
              RGBApixel pixelIm = BMPhelper.getPixelFromValue (imag, maxNegIm, maxIm);
              RGBApixel pixelMod = BMPhelper.getPixelFromValue (mod, maxNegMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

              // Set pixel for current image.
              imageRe.SetPixel(i, j, pixelRe);
#ifdef COMPLEX_FIELD_VALUES
              imageIm.SetPixel(i, j, pixelIm);
              imageMod.SetPixel(i, j, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
            }
          }
        }

        std::string outputRe = output + std::string (".Re.bmp");
        std::string txtoutputRe = output + std::string (".Re.bmp.txt");

        std::string outputIm = output + std::string (".Im.bmp");
        std::string txtoutputIm = output + std::string (".Im.bmp.txt");

        std::string outputMod = output + std::string (".Mod.bmp");
        std::string txtoutputMod = output + std::string (".Mod.bmp.txt");

        imageRe.WriteToFile (outputRe.c_str ());
        fileMaxRe.open (txtoutputRe.c_str (), std::ios::out);
        ASSERT (fileMaxRe.is_open());
        fileMaxRe << std::setprecision(std::numeric_limits<double>::digits10) << maxPosRe << " " << maxNegRe;
        fileMaxRe.close();

#ifdef COMPLEX_FIELD_VALUES
        imageIm.WriteToFile (outputIm.c_str ());
        fileMaxIm.open (txtoutputIm.c_str (), std::ios::out);
        ASSERT (fileMaxIm.is_open());
        fileMaxIm << std::setprecision(std::numeric_limits<double>::digits10) << maxPosIm << " " << maxNegIm;
        fileMaxIm.close();

        imageMod.WriteToFile (outputMod.c_str ());
        fileMaxMod.open (txtoutputMod.c_str (), std::ios::out);
        ASSERT (fileMaxMod.is_open());
        fileMaxMod << std::setprecision(std::numeric_limits<double>::digits10) << maxPosMod << " " << maxNegMod;
        fileMaxMod.close();
#endif /* COMPLEX_FIELD_VALUES */
      }
      else
      {
        ALWAYS_ASSERT (0);
      }
    }

    //
    // Grid<GridCoordinate3D> grid (size, 0, 1, "tmp_grid");
    //
    // std::vector< std::string > fileNames (1);
    // fileNames[0] = input;
    //
    // TXTLoader<GridCoordinate3D> txtLoader3D;
    // txtLoader3D.loadGrid (&grid, start, end, 0, 0, fileNames);
    //
    // BMPDumper<GridCoordinate3D> bmpDumper3D;
    // bmpDumper3D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);
    // bmpDumper3D.dumpGrid (&grid, start, end, 0, 0, 0);
  }
  else
  {
    // TODO: add other dimensions
    ALWAYS_ASSERT (0);
  }



  return 0;
}

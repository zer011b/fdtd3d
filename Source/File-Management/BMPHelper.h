/*
 * Copyright (C) 2016 Gleb Balykov
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

#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldValue.h"
#include "PAssert.h"
#include "Settings.h"

/**
 * Types of palette
 */
ENUM_CLASS (PaletteType, uint8_t,
  PALETTE_BLUE_GREEN_RED, /**< blue - minimum, green - middle, red - maximum */
  PALETTE_GRAY /**< black - minimum, white - maximum */
); /* PaletteType */

/**
 * Types of orthogonal axis
 */
ENUM_CLASS (OrthogonalAxis, uint8_t,
  X,
  Y,
  Z
); /* OrthogonalAxis */

/**
 * Class to statically include in BMPLoader and BMPDumper.
 * Blue-Green-Red scheme is implemented.
 */
class BMPHelper
{
private:

  /**
   * Type of palette
   */
  PaletteType palette;

  /**
   * Type of orthogonal axis
   */
  OrthogonalAxis orthogonalAxis;

public:

  /**
   * Bit depth for BMP image
   */
  static int bitDepth;

  /**
   * Number of degrees of color for BMP image
   */
  static int numColors;

private:

  FPValue getValueFromPixelBlueGreenRed (const RGBApixel &, FPValue, FPValue);
  FPValue getValueFromPixelGray (const RGBApixel &, FPValue, FPValue);
  RGBApixel getPixelFromValueBlueGreenRed (FPValue, FPValue, FPValue);
  RGBApixel getPixelFromValueGray (FPValue, FPValue, FPValue);

public:

  /**
   * Constructor
   */
  BMPHelper (PaletteType colorPalette, /**< color palette */
             OrthogonalAxis orthAxis) /**< orthogonal axis */
  : palette (colorPalette)
  , orthogonalAxis (orthAxis)
  {
  } /* BMPHelper */

  /**
   * Set color palette and orthogonal axis
   */
  void initialize (PaletteType colorPalette, /**< color palette */
                   OrthogonalAxis orthAxis) /**< orthogonal axis */
  {
    palette = colorPalette;
    orthogonalAxis = orthAxis;
  } /* initialize */

  FPValue getValueFromPixel (const RGBApixel &, FPValue, FPValue);
  RGBApixel getPixelFromValue (FPValue, FPValue, FPValue);

  /**
   * Get orthogonal axis
   *
   * @return orthogonal axis
   */
  OrthogonalAxis getOrthogonalAxis ()
  {
    return orthogonalAxis;
  } /* getOrthogonalAxis */
}; /* BMPHelper */

#endif /* BMP_HELPER_H */

#include "Assert.h"
#include "BMPHelper.h"

int BMPHelper::bitDepth = 24;
int BMPHelper::numColors = 255;

/**
 * Return value according to colors of pixel. Blue-Green-Red scheme.
 *
 * @return floating point value of pixel
 */
FPValue
BMPHelper::getValueFromPixelBlueGreenRed (const RGBApixel &pixel, /**< pixel */
                                          FPValue min, /**< minimum value */
                                          FPValue max) /**< maximum value */
{
  FPValue diff = max - min;
  FPValue retval = min;
  FPValue diff_2 = diff / 2.0;

  if (pixel.Blue == 0)
  {
    retval += diff_2 + (((FPValue) pixel.Red) / BMPHelper::numColors) * diff_2;
  }
  else
  {
    ASSERT (pixel.Red == 0);
    retval += (((FPValue) pixel.Green) / BMPHelper::numColors) * diff_2;
  }

  return retval;
} /* BMPHelper::getValueFromPixelBlueGreenRed */

/**
 * Return value according to colors of pixel. Gray scheme.
 *
 * @return floating point value of pixel
 */
FPValue
BMPHelper::getValueFromPixelGray (const RGBApixel &pixel, /**< pixel */
                                  FPValue min, /**< minimum value */
                                  FPValue max) /**< maximum value */
{
  ASSERT (pixel.Red == pixel.Green
          && pixel.Red == pixel.Blue);

  FPValue diff = max - min;
  FPValue retval = min + (((FPValue) pixel.Red) / BMPHelper::numColors) * diff;

  return retval;
} /* BMPHelper::getValueFromPixelGray */

/**
 * Return pixel with colors according to values. Blue-Green-Red scheme.
 *
 * @return pixel
 */
RGBApixel
BMPHelper::getPixelFromValueBlueGreenRed (FPValue val, /**< value */
                                          FPValue min, /**< minimum value */
                                          FPValue max) /**< maximum value */
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FPValue diff = max - min;
  FPValue value = val - min;
  FPValue diff_2 = diff / 2.0;
  if (value > diff_2)
  {
    value -= diff_2;
    FPValue tmp = value / diff_2;
    pixel.Red = tmp * BMPHelper::numColors;
    pixel.Green = (1.0 - tmp) * BMPHelper::numColors;
    pixel.Blue = 0.0;
  }
  else
  {
    FPValue tmp = 0;
    if (diff == 0)
    {
      tmp = 0.0;
    }
    else
    {
      tmp = value / diff_2;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * BMPHelper::numColors;
    pixel.Blue = (1.0 - tmp) * BMPHelper::numColors;
  }

  return pixel;
} /* BMPHelper::getPixelFromValueBlueGreenRed */

/**
 * Return pixel with colors according to values. Gray scheme.
 *
 * @return pixel
 */
RGBApixel
BMPHelper::getPixelFromValueGray (FPValue val, /**< value */
                                  FPValue min, /**< minimum value */
                                  FPValue max) /**< maximum value */
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FPValue diff = max - min;
  FPValue value = val - min;
  pixel.Red = value / diff * BMPHelper::numColors;
  pixel.Green = value / diff * BMPHelper::numColors;
  pixel.Blue = value / diff * BMPHelper::numColors;

  return pixel;
} /* BMPHelper::getPixelFromValueGray */

/**
 * Return value according to colors of pixel
 *
 * @return floating point value of pixel
 */
FPValue
BMPHelper::getValueFromPixel (const RGBApixel &pixel, /**< pixel */
                              FPValue min, /**< minimum value */
                              FPValue max) /**< maximum value */
{
  switch (palette)
  {
    case PaletteType::PALETTE_BLUE_GREEN_RED:
    {
      return getValueFromPixelBlueGreenRed (pixel, min, max);
    }
    case PaletteType::PALETTE_GRAY:
    {
      return getValueFromPixelGray (pixel, min, max);
    }
    default:
    {
      UNREACHABLE;
      return FPValue (0);
    }
  }
} /* BMPHelper::getValueFromPixel */

/**
 * Return pixel with colors according to values
 *
 * @return pixel
 */
RGBApixel
BMPHelper::getPixelFromValue (FPValue val, /**< value */
                              FPValue min, /**< minimum value */
                              FPValue max) /**< maximum value */
{
  switch (palette)
  {
    case PaletteType::PALETTE_BLUE_GREEN_RED:
    {
      return getPixelFromValueBlueGreenRed (val, min, max);
    }
    case PaletteType::PALETTE_GRAY:
    {
      return getPixelFromValueGray (val, max, max);
    }
    default:
    {
      UNREACHABLE;
      return RGBApixel ();
    }
  }
} /* BMPHelper::getPixelFromValue */

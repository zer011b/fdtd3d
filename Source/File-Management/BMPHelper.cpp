#include "Assert.h"
#include "BMPHelper.h"

int BMPHelper::bitDepth = 24;
int BMPHelper::numColors = 256;

/**
 * Return value according to colors of pixel.
 * Blue-Green-Red scheme.
 */
FPValue
BMPHelper::getValueFromPixelBlueGreenRed (const RGBApixel& pixel, const FPValue& maxNeg,
                                          const FPValue& max)
{
  /*
   * FIXME: use maxNeg
   */
  FPValue retval = 0;
  FPValue max_2 = max / 2.0;

  if (pixel.Blue == 0 && pixel.Red == 0)
  {
    retval = max_2;
  }
  else if (pixel.Blue == 0)
  {
    retval = (((FPValue) pixel.Red) / BMPHelper::numColors + 1) * max_2;
  }
  else if (pixel.Red == 0)
  {
    retval = (((FPValue) pixel.Green) / BMPHelper::numColors) * max_2;
  }
  else
  {
    UNREACHABLE;
  }

  return retval;
}

/**
 * Return value according to colors of pixel.
 * Gray scheme.
 */
FPValue
BMPHelper::getValueFromPixelGray (const RGBApixel& pixel, const FPValue& maxNeg,
                                  const FPValue& max)
{
  /*
   * FIXME: use maxNeg
   */
  ASSERT (pixel.Red == pixel.Green
          && pixel.Red == pixel.Blue);

  FPValue retval = ((FPValue) pixel.Red) / BMPHelper::numColors * max;

  return retval;
}

/**
 * Return pixel with colors according to values.
 * Blue-Green-Red scheme.
 */
RGBApixel
BMPHelper::getPixelFromValueBlueGreenRed (const FPValue& val, const FPValue& maxNeg,
                                          const FPValue& max)
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FPValue value = val - maxNeg;
  FPValue max_2 = max / 2.0;
  if (value > max_2)
  {
    value -= max_2;
    FPValue tmp = value / max_2;
    pixel.Red = tmp * BMPHelper::numColors;
    pixel.Green = (1.0 - tmp) * BMPHelper::numColors;
    pixel.Blue = 0.0;
  }
  else
  {
    FPValue tmp = 0;
    if (max == 0)
    {
      tmp = 0.0;
    }
    else
    {
      tmp = value / max_2;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * BMPHelper::numColors;
    pixel.Blue = (1.0 - tmp) * BMPHelper::numColors;
  }

  return pixel;
}

/**
 * Return pixel with colors according to values.
 * Gray scheme.
 */
RGBApixel
BMPHelper::getPixelFromValueGray (const FPValue& val, const FPValue& maxNeg,
                                  const FPValue& max)
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FPValue value = val - maxNeg;
  pixel.Red = value / max * BMPHelper::numColors;
  pixel.Green = value / max * BMPHelper::numColors;
  pixel.Blue = value / max * BMPHelper::numColors;

  return pixel;
}

FPValue
BMPHelper::getValueFromPixel (const RGBApixel& pixel, const FPValue& maxNeg,
                              const FPValue& max)
{
  switch (palette)
  {
    case PaletteType::PALETTE_BLUE_GREEN_RED:
    {
      return getValueFromPixelBlueGreenRed (pixel, maxNeg, max);

      break;
    }
    case PaletteType::PALETTE_GRAY:
    {
      return getValueFromPixelGray (pixel, maxNeg, max);

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

RGBApixel
BMPHelper::getPixelFromValue (const FPValue& val, const FPValue& maxNeg,
                              const FPValue& max)
{
  switch (palette)
  {
    case PaletteType::PALETTE_BLUE_GREEN_RED:
    {
      return getPixelFromValueBlueGreenRed (val, maxNeg, max);

      break;
    }
    case PaletteType::PALETTE_GRAY:
    {
      return getPixelFromValueGray (val, maxNeg, max);

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

#include "BMPHelper.h"

/*
 * Return value according to colors of pixel.
 * Blue-Green-Red scheme.
 */
FieldValue
BMPHelper::getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                              const FieldValue& max)
{
  FieldValue retval = 0;
  FieldValue max_2 = max / 2.0;

  if (pixel.Blue == 0 && pixel.Red == 0)
  {
    retval = max_2;
  }
  else if (pixel.Blue == 0)
  {
    retval = (((FieldValue) pixel.Red) / 255 + 1) * max_2;
  }
  else if (pixel.Red == 0)
  {
    retval = (((FieldValue) pixel.Green) / 255) * max_2;
  }
  else
  {
    UNREACHABLE;
  }

  return retval;
}

/*
 * Return pixel with colors according to values.
 * Blue-Green-Red scheme.
 */
RGBApixel
BMPHelper::getPixelFromValue (const FieldValue& val, const FieldValue& maxNeg,
                              const FieldValue& max)
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  FieldValue value = val - maxNeg;
  FieldValue max_2 = max / 2.0;
  if (value > max_2)
  {
    value -= max_2;
    FieldValue tmp = value / max_2;
    pixel.Red = tmp * 255;
    pixel.Green = (1.0 - tmp) * 255;
    pixel.Blue = 0.0;
  }
  else
  {
    FieldValue tmp = 0;
    if (max == 0)
    {
      tmp = 0.0;
    }
    else
    {
      tmp = value / max_2;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * 255;
    pixel.Blue = (1.0 - tmp) * 255;
  }

  return pixel;
}

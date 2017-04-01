#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldPoint.h"

#ifdef CXX11_ENABLED

enum class PaletteType: uint32_t
{
  PALETTE_BLUE_GREEN_RED,
  PALETTE_GRAY
};

enum class OrthogonalAxis: uint32_t
{
  X,
  Y,
  Z
};

#else /* CXX11_ENABLED */

class PaletteType
{
public:
  enum PaletteTypeEnum
  {
    PALETTE_BLUE_GREEN_RED,
    PALETTE_GRAY
  };

  PaletteType (PaletteTypeEnum new_type)
    : paletteType (new_type)
  {
  }

  operator int ()
  {
    return paletteType;
  }

private:

  PaletteTypeEnum paletteType;
};

class OrthogonalAxis
{
public:
  enum OrthogonalAxisEnum
  {
    X,
    Y,
    Z
  };

  OrthogonalAxis (OrthogonalAxisEnum new_axis)
    : orthogonalAxis (new_axis)
  {
  }

  operator int ()
  {
    return orthogonalAxis;
  }

private:

  OrthogonalAxisEnum orthogonalAxis;
};

#endif /* !CXX11_ENABLED */

/**
 * Class to statically include in BMPLoader and BMPDumper.
 * Blue-Green-Red scheme is implemented.
 */
class BMPHelper
{
private:

  PaletteType palette;

  OrthogonalAxis orthogonalAxis;

private:

  FPValue getValueFromPixelBlueGreenRed (const RGBApixel& pixel, const FPValue& maxNeg,
                                         const FPValue& max);
  FPValue getValueFromPixelGray (const RGBApixel& pixel, const FPValue& maxNeg,
                                 const FPValue& max);

  RGBApixel getPixelFromValueBlueGreenRed (const FPValue& value, const FPValue& maxNeg,
                                           const FPValue& max);

  RGBApixel getPixelFromValueGray (const FPValue& value, const FPValue& maxNeg,
                                   const FPValue& max);

public:

  BMPHelper (PaletteType colorPalette,
             OrthogonalAxis orthAxis)
    : palette (colorPalette)
  , orthogonalAxis (orthAxis)
  {
  }

  // Return value with values according to colors of pixel.
  FPValue getValueFromPixel (const RGBApixel& pixel, const FPValue& maxNeg,
                             const FPValue& max);

  // Return pixel with colors according to values of value.
  RGBApixel getPixelFromValue (const FPValue& value, const FPValue& maxNeg,
                               const FPValue& max);

  OrthogonalAxis getOrthogonalAxis ()
  {
    return orthogonalAxis;
  }
};

#endif /* BMP_HELPER_H */

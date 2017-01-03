#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldPoint.h"

enum class PaletteType: uint32_t
{
  PALETTE_BLUE_GREEN_RED,
  PALETTE_GRAY
};

/**
 * Class to statically include in BMPLoader and BMPDumper.
 * Blue-Green-Red scheme is implemented.
 */
class BMPHelper
{
private:

  PaletteType palette;

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

  BMPHelper (PaletteType colorPalette)
    : palette (colorPalette)
  {
  }

  // Return value with values according to colors of pixel.
  FPValue getValueFromPixel (const RGBApixel& pixel, const FPValue& maxNeg,
                             const FPValue& max);

  // Return pixel with colors according to values of value.
  RGBApixel getPixelFromValue (const FPValue& value, const FPValue& maxNeg,
                               const FPValue& max);
};

#endif /* BMP_HELPER_H */

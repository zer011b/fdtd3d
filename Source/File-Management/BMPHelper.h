#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldValue.h"
#include "Assert.h"
#include "Settings.h"

ENUM_CLASS (PaletteType, uint8_t,
  PALETTE_BLUE_GREEN_RED,
  PALETTE_GRAY
);

ENUM_CLASS (OrthogonalAxis, uint8_t,
  X,
  Y,
  Z
);

/**
 * Class to statically include in BMPLoader and BMPDumper.
 * Blue-Green-Red scheme is implemented.
 */
class BMPHelper
{
private:

  PaletteType palette;

  OrthogonalAxis orthogonalAxis;

public:

  static int bitDepth;
  static int numColors;

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

  void initialize (PaletteType colorPalette,
                   OrthogonalAxis orthAxis)
  {
    palette = colorPalette;
    orthogonalAxis = orthAxis;
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

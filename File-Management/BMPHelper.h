#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldPoint.h"

/**
 * Class to statically include in BMPLoader and BMPDumper.
 * Blue-Green-Red scheme is implemented.
 */
class BMPHelper
{
public:

  // Return value with values according to colors of pixel.
  static FieldValue getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                                       const FieldValue& max);

  // Return pixel with colors according to values of value.
  static RGBApixel getPixelFromValue (const FieldValue& value, const FieldValue& maxNeg,
                                      const FieldValue& max);
};

#endif /* BMP_HELPER_H */

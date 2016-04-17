#ifndef BMP_HELPER_H
#define BMP_HELPER_H

#include "EasyBMP.h"
#include "FieldPoint.h"

class BMPHelper
{
public:
  // Return pixel with colors according to values.
  static FieldValue getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                                       const FieldValue& max);

  // Return pixel with colors according to values.
  static RGBApixel getPixelFromValue (const FieldValue& value, const FieldValue& maxNeg,
                                      const FieldValue& max);
};

#endif /* BMP_HELPER_H */

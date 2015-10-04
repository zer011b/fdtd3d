#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "Dumper.h"
#include "EasyBMP.h"

class BMPDumper: public Dumper
{

private:

#if defined (GRID_1D)
  void dump1D (Grid& grid);
#else
#if defined (GRID_2D)
  void dump2D (Grid& grid);
#else
#if defined (GRID_3D)
  void dump3D (Grid& grid);
#endif
#endif
#endif

  RGBApixel setPixel (const FieldValue& value, const FieldValue& maxNeg,
                      const FieldValue& max);

public:

  void dumpGrid (Grid& grid) override;
};

#endif /* BMP_DUMPER_H */
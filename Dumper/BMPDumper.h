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

  /*
   * Sets pixel colors
   */
  RGBApixel setPixel (const FieldValue& value, const FieldValue& maxNeg,
                      const FieldValue& max);

  /*
   * Dumps flat grids: 1D and 2D
   */
  void dumpFlat (Grid& grid, grid_iter sx, grid_iter sy);

public:

  void dumpGrid (Grid& grid) override;
};

#endif /* BMP_DUMPER_H */
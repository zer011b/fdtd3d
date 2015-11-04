#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "Dumper.h"
#include "EasyBMP.h"

class BMPDumper: public Dumper
{
private:

#if defined (GRID_1D)
  void dump1D (Grid& grid) const;
#else
#if defined (GRID_2D)
  void dump2D (Grid& grid) const;
#else
#if defined (GRID_3D)
  void dump3D (Grid& grid) const;
#endif
#endif
#endif

  /*
   * Return pixel with colors according to values
   */
  RGBApixel getPixel (const FieldValue& value, const FieldValue& maxNeg,
                      const FieldValue& max) const;

  /*
   * Dumps flat grids: 1D and 2D
   */
  void dumpFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const;

public:

  /**
   * Dump function to call for every grid type
   */
  void dumpGrid (Grid& grid) const override;
};

#endif /* BMP_DUMPER_H */

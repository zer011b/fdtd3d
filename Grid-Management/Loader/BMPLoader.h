#ifndef BMPLOADER_H
#define BMPLOADER_H

#include "Loader.h"
#include "EasyBMP.h"

class BMPLoader: public Loader
{
private:

#if defined (GRID_1D)
  void load1D (Grid& grid) const;
#else
#if defined (GRID_2D)
  void load2D (Grid& grid) const;
#else
#if defined (GRID_3D)
  void load3D (Grid& grid) const;
#endif
#endif
#endif

  /*
   * Return pixel with colors according to values
   */
  FieldValue getValueFromPixel (const RGBApixel& pixel, const FieldValue& maxNeg,
                                const FieldValue& max) const;

  void loadFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const;

public:

  void LoadGrid (Grid& grid) const override;
};

#endif /* BMPLOADER_H */

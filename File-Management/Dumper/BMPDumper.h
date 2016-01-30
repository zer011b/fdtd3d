#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "Dumper.h"
#include "EasyBMP.h"

// Grid saver in BMP files.
class BMPDumper: public Dumper
{
private:

#if defined (GRID_1D)
  // Save one-dimensional grid.
  void dump1D (Grid& grid) const;
#else /* GRID_1D */
#if defined (GRID_2D)
  // Save two-dimensional grid.
  void dump2D (Grid& grid) const;
#else /* GRID_2D */
#if defined (GRID_3D)
  // Save three-dimensional grid.
  void dump3D (Grid& grid) const;
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */

  // Return pixel with colors according to values.
  RGBApixel getPixelFromValue (const FieldValue& value, const FieldValue& maxNeg,
                               const FieldValue& max) const;

  // Save flat grids: 1D and 2D.
  void dumpFlat (Grid& grid, const grid_iter& sx, const grid_iter& sy) const;

  void writeToFile (Grid& grid, const grid_iter& sx, const grid_iter& sy, GridFileType dump_type) const;

public:

  // Function to call for every grid type.
  void dumpGrid (Grid& grid) const override;
};

#endif /* BMP_DUMPER_H */

#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "BMPHelper.h"
#include "Dumper.h"

/**
 * Grid saver to BMP files.
 * Template class with coordinate parameter.
 */
template <class TCoord>
class BMPDumper: public Dumper<TCoord>
{
  // Helper class for usage with BMP files.
  static BMPHelper BMPhelper;

private:

  // Save grid to file for specific layer.
  void writeToFile (Grid<TCoord> &grid, GridFileType dump_type) const;

  // Save grid to file for all layers.
  void writeToFile (Grid<TCoord> &grid) const;

public:

  virtual ~BMPDumper () {}

  // Virtual method for grid loading.
  virtual void dumpGrid (Grid<TCoord> &grid) const override;
};

/**
 * ======== Template implementation ========
 */

/**
 * Save grid to file for all layers.
 */
template <class TCoord>
void
BMPDumper<TCoord>::writeToFile (Grid<TCoord> &grid) const
{
  writeToFile (grid, CURRENT);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS);
  }
#if defined (TWO_TIME_STEPS)
  if (GridFileManager::type == ALL)
  {
    writeToFile (grid, PREVIOUS2);
  }
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
}

#endif /* BMP_DUMPER_H */

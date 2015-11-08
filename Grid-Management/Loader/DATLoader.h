#ifndef DAT_LOADER_H
#define DAT_LOADER_H

#include "Loader.h"

// Grid loader from binary files.
class DATLoader: public Loader
{
  void loadFromFile (Grid& grid, GridFileType type) const;

public:

  // Function to call for every grid type.
  void loadGrid (Grid& grid) const override;
};

#endif /* DAT_LOADER_H */

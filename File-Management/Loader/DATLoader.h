#ifndef DAT_LOADER_H
#define DAT_LOADER_H

#include "Loader.h"

// Grid loader from binary files.
template <class TGrid>
class DATLoader: public Loader<TGrid>
{
  void loadFromFile (TGrid& grid, GridFileType type) const;

public:

  // Function to call for every grid type.
  void loadGrid (TGrid& grid) const override;
};

#endif /* DAT_LOADER_H */

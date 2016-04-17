#ifndef DAT_DUMPER_H
#define DAT_DUMPER_H

#include "Dumper.h"

// Grid saver in binary files.
template <class TGrid>
class DATDumper: public Dumper<TGrid>
{
  void writeToFile (TGrid& grid, GridFileType type) const;

public:

  // Function to call for every grid type.
  void dumpGrid (TGrid& grid) const override;
};

#endif /* DAT_DUMPER_H */

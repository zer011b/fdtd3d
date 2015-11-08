#ifndef DAT_DUMPER_H
#define DAT_DUMPER_H

#include "Dumper.h"

// Grid saver in binary files.
class DATDumper: public Dumper
{
  void writeToFile (Grid& grid, GridFileType type) const;

public:

  // Function to call for every grid type.
  void dumpGrid (Grid& grid) const override;
};

#endif /* DAT_DUMPER_H */

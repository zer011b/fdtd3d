#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "Dumper.h"
#include "BMPHelper.h"

// Grid saver in BMP files.
template <class TGrid>
class BMPDumper: public Dumper<TGrid>
{
  static BMPHelper BMPhelper;

private:

  void writeToFile (TGrid& grid, GridFileType dump_type);
  void writeToFile (TGrid& grid);

public:

  virtual ~BMPDumper () {}

  // Function to call for every grid type.
  void dumpGrid (TGrid& grid) const override;
};

#endif /* BMP_DUMPER_H */

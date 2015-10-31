#ifndef BMPLOADER_H
#define BMPLOADER_H

#include "Loader.h"
#include "EasyBMP.h"

class BMPLoader: public Loader
{
public:
  BMPLoader();

  void LoadGrid (Grid& grid) const override {}
  void init (const grid_iter& timeStep, GridFileType newType) override {}
  void setStep (const grid_iter& timeStep) override {}
  void setGridFileType (GridFileType newType) override {}
};

#endif /* BMPLOADER_H */

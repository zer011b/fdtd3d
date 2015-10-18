#ifndef DUMPER_H
#define DUMPER_H

#include <string>

#include "FieldGrid.h"
#include "commons.h"

class Dumper
{
protected:
  grid_iter step;
  GridFileType type;

  std::string cur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  std::string prev;
#if defined (TWO_TIME_STEPS)
  std::string prevPrev;
#endif
#endif

public:

  Dumper () : step (0), type (ALL)
  {
  }

  virtual void dumpGrid (Grid& grid) const = 0;
  virtual void init (const grid_iter& timeStep, GridFileType newType) = 0;
  virtual void setStep (const grid_iter& timeStep) = 0;
  virtual void setGridFileType (GridFileType newType) = 0;
};

#endif /* DUMPER_H */

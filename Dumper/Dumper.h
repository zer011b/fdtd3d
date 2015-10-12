#ifndef DUMPER_H
#define DUMPER_H

#include <string>

#include "FieldGrid.h"

enum DumpType
{
  DUMP_CURRENT,
  DUMP_ALL
};


class Dumper
{
protected:
  grid_iter step;
  DumpType type;

  std::string cur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  std::string prev;
#if defined (TWO_TIME_STEPS)
  std::string prevPrev;
#endif
#endif

public:

  Dumper () : step (0), type (DUMP_ALL)
  {
  }

  virtual void dumpGrid (Grid& grid) const = 0;
  virtual void init (const grid_iter& timeStep, DumpType newType) = 0;
  virtual void setStep (const grid_iter& timeStep) = 0;
  virtual void setDumpType (DumpType newType) = 0;
};

#endif /* DUMPER_H */
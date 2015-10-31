#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "FieldGrid.h"

enum GridFileType
{
  CURRENT,
  ALL
};

class GridFileManager
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

  GridFileManager () : step (0), type (ALL)
  {
  }

  virtual void init (const grid_iter& timeStep, GridFileType newType) = 0;
  virtual void setStep (const grid_iter& timeStep) = 0;
  virtual void setGridFileType (GridFileType newType) = 0;
};

#endif /* COMMONS_H */

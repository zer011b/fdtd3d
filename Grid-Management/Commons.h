#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "FieldGrid.h"
#include "Assert.h"

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

  void setFileNames ()
  {
    cur.clear ();
    cur = std::string ("current[") + std::to_string (step) + std::string ("].bmp");
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev.clear ();
    prev = std::string ("previous[") + std::to_string (step) + std::string ("].bmp");
#if defined (TWO_TIME_STEPS)
    prevPrev.clear ();
    prevPrev = std::string ("previous2[") + std::to_string (step) + std::string ("].bmp");
#endif
#endif
  }

public:

  GridFileManager () : step (0), type (ALL)
  {
  }

  void init (const grid_iter& timeStep, GridFileType newType)
  {
    step = timeStep;
    type = newType;

    setFileNames ();
  }

  void setStep (const grid_iter& timeStep)
  {
    step = timeStep;

    setFileNames();
  }

  void setGridFileType (GridFileType newType)
  {
    type = newType;
  }
};

#endif /* COMMONS_H */

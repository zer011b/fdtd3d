#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "FieldGrid.h"
#include "Assert.h"

// Type of save/load from file.
// CURRENT: save/load only current layer.
// ALL: save/load all layers.
enum GridFileType
{
  CURRENT,
  PREVIOUS,
  PREVIOUS2,
  ALL
};

// Basic class for all dumpers/loaders.
class GridFileManager
{
protected:
  // Time step number (used in file naming).
  grid_iter step;

  // Save/load type.
  GridFileType type;

  // File names.
  std::string cur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  std::string prev;
#if defined (TWO_TIME_STEPS)
  std::string prevPrev;
#endif
#endif

  // Set file names according to time step.
  void setFileNames ()
  {
    cur.clear ();
    cur = std::string ("current[") + std::to_string (step) + std::string ("]");
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev.clear ();
    prev = std::string ("previous[") + std::to_string (step) + std::string ("]");
#if defined (TWO_TIME_STEPS)
    prevPrev.clear ();
    prevPrev = std::string ("previous2[") + std::to_string (step) + std::string ("]");
#endif
#endif
  }

public:

  GridFileManager () : step (0), type (ALL)
  {
  }

  // Initialize dumper with time step number and save/load type.
  void init (const grid_iter& timeStep, GridFileType newType)
  {
    step = timeStep;
    type = newType;

    setFileNames ();
  }

  // Set time step.
  void setStep (const grid_iter& timeStep)
  {
    step = timeStep;

    setFileNames();
  }

  // Set save/load type.
  void setGridFileType (GridFileType newType)
  {
    type = newType;
  }
};

#endif /* COMMONS_H */

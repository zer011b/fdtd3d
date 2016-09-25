#ifndef COMMONS_H
#define COMMONS_H

#include <string>

#include "Assert.h"
#include "Grid.h"

extern std::string int64_to_string(int64_t value);

/**
 * Type of save/load from file.
 * CURRENT: current layer.
 * PREVIOUS: previous layer.
 * PREVIOUS2: previous to previous layer.
 * ALL: all layers.
 */
enum GridFileType
{
  CURRENT,
  PREVIOUS,
  PREVIOUS2,
  ALL
};

/**
 * Base class for all dumpers/loaders.
 */
class GridFileManager
{
protected:

  // Time step number (used in file naming).
  grid_iter step;

  // Save/load type.
  GridFileType type;

  int processId;

  // File names.
  // File name for current layer file.
  std::string cur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // File name for previous layer file.
  std::string prev;
#if defined (TWO_TIME_STEPS)
  // File name for previous to previous layer file.
  std::string prevPrev;
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

  // Set file names according to time step.
  void setFileNames ()
  {
    cur.clear ();
    cur = std::string ("current[") + int64_to_string (step) + std::string ("]_rank-") + int64_to_string (processId);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev.clear ();
    prev = std::string ("previous[") + int64_to_string (step) + std::string ("]_rank-") + int64_to_string (processId);
#if defined (TWO_TIME_STEPS)
    prevPrev.clear ();
    prevPrev = std::string ("previous2[") + int64_to_string (step) + std::string ("]_rank-") + int64_to_string (processId);
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  }

  // Protected constructor to disallow instantiation.
  GridFileManager () : step (0), type (ALL) {}

public:

  virtual ~GridFileManager () {}

  // Initialize dumper with time step number and save/load type.
  void init (const grid_iter& timeStep, GridFileType newType, int process)
  {
    step = timeStep;
    type = newType;
    processId = process;

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

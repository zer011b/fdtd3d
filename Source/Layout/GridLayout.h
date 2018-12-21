#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include "GridCoordinate3D.h"

/**
 * Direction in which to get circut elements
 */
ENUM_CLASS (LayoutDirection, uint8_t,
  LEFT, /**< left by Ox */
  RIGHT, /**< right by Ox */
  DOWN, /**< left by Oy */
  UP, /**< right by Oy */
  BACK, /**< left by Oz */
  FRONT /**< right by Oz */
);

/**
 * Type of electromagnetic field.
 */
ENUM_CLASS (GridType, uint8_t,
  NONE,
  EX,
  EY,
  EZ,
  HX,
  HY,
  HZ,
  DX,
  DY,
  DZ,
  BX,
  BY,
  BZ,
  EPS,
  MU,
  SIGMAX,
  SIGMAY,
  SIGMAZ,
  OMEGAPE,
  OMEGAPM,
  GAMMAE,
  GAMMAM
);

#endif /* GRID_LAYOUT_H */

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

/**
 * Type of field, which is placed at the center of the grid cell
 *
 * E_CENTERED:
 *
 *             Ex is:
 *                   1 <= x < 1 + size.getx()
 *                   0.5 <= y < 0.5 + size.getY()
 *                   0.5 <= z < 0.5 + size.getZ()
 *             Ey is:
 *                   0.5 <= x < 0.5 + size.getx()
 *                   1 <= y < 1 + size.getY()
 *                   0.5 <= z < 0.5 + size.getZ()
 *             Ez is:
 *                   0.5 <= x < 0.5 + size.getx()
 *                   0.5 <= y < 0.5 + size.getY()
 *                   1 <= z < 1 + size.getZ()
 *             Hx is:
 *                   0.5 <= x < 0.5 + size.getx()
 *                   1 <= y < 1 + size.getY()
 *                   1 <= z < 1 + size.getZ()
 *             Hy is:
 *                   1 <= x < 1 + size.getx()
 *                   0.5 <= y < 0.5 + size.getY()
 *                   1 <= z < 1 + size.getZ()
 *             Hz is:
 *                   1 <= z < 1 + size.getx()
 *                   1 <= y < 1 + size.getY()
 *                   0.5 <= z < 0.5 + size.getZ()
 *
 * H_CENTERED:
 * TODO: add description
 */
enum LayoutType
{
  E_CENTERED,
  H_CENTERED
};

#endif /* GRID_LAYOUT_H */

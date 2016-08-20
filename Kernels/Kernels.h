#ifndef KERNELS_H
#define KERNELS_H

#include "FieldValue.h"

/* 2D TMz */
#define calculateEz_2D_TMz(oldEz, Hx1, Hx2, Hy1, Hy2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Hx1) - (Hx2) + (Hy1) - (Hy2)))

#define calculateHx_2D_TMz(oldHx, Ez1, Ez2, dt, dx, mu) \
  ((oldHx) + ((dt) / ((mu) * (dx))) * ((Ez1) - (Ez2)))

#define calculateHy_2D_TMz(oldHy, Ez1, Ez2, dt, dx, mu) \
  calculateHx_2D_TMz ((oldHy), (Ez1), (Ez2), (dt), (dx), (mu))


/* 3D */
#define calculateEx_3D(oldEx, Hz1, Hz2, Hy1, Hy2, dt, dx, eps) \
  ((oldEx) + ((dt) / ((eps) * (dx))) * ((Hz1) - (Hz2) - (Hy1) + (Hy2)))

#define calculateEy_3D(oldEy, Hx1, Hx2, Hz1, Hz2, dt, dx, eps) \
  ((oldEy) + ((dt) / ((eps) * (dx))) * ((Hx1) - (Hx2) - (Hz1) + (Hz2)))

#define calculateEz_3D(oldEz, Hy1, Hy2, Hx1, Hx2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Hy1) - (Hy2) - (Hx1) + (Hx2)))

#define calculateHx_3D(oldHx, Ey1, Ey2, Ez1, Ez2, dt, dx, mu) \
  ((oldHx) + ((dt) / ((mu) * (dx))) * ((Ey1) - (Ey2) - (Ez1) + (Ez2)))

#define calculateHy_3D(oldHy, Ez1, Ez2, Ex1, Ex2, dt, dx, mu) \
  ((oldHy) + ((dt) / ((mu) * (dx))) * ((Ez1) - (Ez2) - (Ex1) + (Ex2)))

#define calculateHz_3D(oldHz, Ex1, Ex2, Ey1, Ey2, dt, dx, mu) \
  ((oldHz) + ((dt) / ((mu) * (dx))) * ((Ex1) - (Ex2) - (Ey1) + (Ey2)))

#endif /* KERNELS_H */

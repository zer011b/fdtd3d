#ifndef KERNELS_H
#define KERNELS_H

#include "FieldValue.h"

/*
 * For all kernels index 1 represents field value with higher coordinate than index 2.
 *
 * FIXME: add docs link to formulas description
 */

/* 3D Kernels with precalculated coefficients */
#define calculateEx_3D_Precalc(oldEx, Hz1, Hz2, Hy1, Hy2, Ca, Cb) \
  ((Ca) * (oldEx) + (Cb) * ((Hz1) - (Hz2) - (Hy1) + (Hy2)))

#define calculateEy_3D_Precalc(oldEy, Hx1, Hx2, Hz1, Hz2, Ca, Cb) \
  ((Ca) * (oldEy) + (Cb) * ((Hx1) - (Hx2) - (Hz1) + (Hz2)))

#define calculateEz_3D_Precalc(oldEz, Hy1, Hy2, Hx1, Hx2, Ca, Cb) \
  ((Ca) * (oldEz) + (Cb) * ((Hy1) - (Hy2) - (Hx1) + (Hx2)))

#define calculateHx_3D_Precalc(oldHx, Ey1, Ey2, Ez1, Ez2, Da, Db) \
  ((Da) * (oldHx) + (Db) * ((Ey1) - (Ey2) - (Ez1) + (Ez2)))

#define calculateHy_3D_Precalc(oldHy, Ez1, Ez2, Ex1, Ex2, Da, Db) \
  ((Da) * (oldHy) + (Db) * ((Ez1) - (Ez2) - (Ex1) + (Ex2)))

#define calculateHz_3D_Precalc(oldHz, Ex1, Ex2, Ey1, Ey2, Da, Db) \
  ((Da) * (oldHz) + (Db) * ((Ex1) - (Ex2) - (Ey1) + (Ey2)))

/* 2D Kernels with precalculated coefficients */
#define calculateEx_2D_TEz_Precalc(oldEx, Hz1, Hz2, Ca, Cb) \
  calculateEx_3D_Precalc(oldEx, Hz1, Hz2, 0, 0, Ca, Cb)

#define calculateEy_2D_TEz_Precalc(oldEy, Hz1, Hz2, Ca, Cb) \
  calculateEy_3D_Precalc(oldEy, 0, 0, Hz1, Hz2, Ca, Cb)

#define calculateHx_2D_TMz_Precalc(oldHx, Ez1, Ez2, Da, Db) \
  calculateHx_3D_Precalc(oldHx, 0, 0, Ez1, Ez2, Da, Db)

#define calculateHy_2D_TMz_Precalc(oldHy, Ez1, Ez2, Da, Db) \
  calculateHy_3D_Precalc(oldHy, Ez1, Ez2, 0, 0, Da, Db)

/* 3D Kernels */
#define calculateEx_3D(oldEx, Hz1, Hz2, Hy1, Hy2, dt, dx, eps) \
  calculateEx_3D_Precalc(oldEx, Hz1, Hz2, Hy1, Hy2, 1, (dt) / ((eps) * (dx)))

#define calculateEy_3D(oldEy, Hx1, Hx2, Hz1, Hz2, dt, dx, eps) \
  calculateEy_3D_Precalc(oldEy, Hx1, Hx2, Hz1, Hz2, 1, (dt) / ((eps) * (dx)))

#define calculateEz_3D(oldEz, Hy1, Hy2, Hx1, Hx2, dt, dx, eps) \
  calculateEz_3D_Precalc(oldEz, Hy1, Hy2, Hx1, Hx2, 1, (dt) / ((eps) * (dx)))

#define calculateHx_3D(oldHx, Ey1, Ey2, Ez1, Ez2, dt, dx, mu) \
  calculateHx_3D_Precalc(oldHx, Ey1, Ey2, Ez1, Ez2, 1, (dt) / ((mu) * (dx)))

#define calculateHy_3D(oldHy, Ez1, Ez2, Ex1, Ex2, dt, dx, mu) \
  calculateHy_3D_Precalc(oldHy, Ez1, Ez2, Ex1, Ex2, 1, (dt) / ((mu) * (dx)))

#define calculateHz_3D(oldHz, Ex1, Ex2, Ey1, Ey2, dt, dx, mu) \
  calculateHz_3D_Precalc(oldHz, Ex1, Ex2, Ey1, Ey2, 1, (dt) / ((mu) * (dx)))

/* 2D Kernels */
#define calculateEx_2D_TEz(oldEx, Hz1, Hz2, dt, dx, eps) \
  calculateEx_3D(oldEx, Hz1, Hz2, 0, 0, dt, dx, eps)

#define calculateEy_2D_TEz(oldEy, Hz1, Hz2, dt, dx, eps) \
  calculateEy_3D(oldEy, 0, 0, Hz1, Hz2, dt, dx, eps)

#define calculateHx_2D_TMz(oldHx, Ez1, Ez2, dt, dx, mu) \
  calculateHx_3D(oldHx, 0, 0, Ez1, Ez2, dt, dx, mu)

#define calculateHy_2D_TMz(oldHy, Ez1, Ez2, dt, dx, mu) \
  calculateHy_3D(oldHy, Ez1, Ez2, 0, 0, dt, dx, mu)

/* Kernels to calculate E from D and H from B */
#define calculateEx_from_Dx_Precalc(oldEx, Dx1, Dx2, Ca, Cb, Cc) \
  ((Ca) * (oldEx) + (Cb) * (Dx1) - (Cc) * (Dx2))

#define calculateEy_from_Dy_Precalc(oldEy, Dy1, Dy2, Ca, Cb, Cc) \
  ((Ca) * (oldEy) + (Cb) * (Dy1) - (Cc) * (Dy2))

#define calculateEz_from_Dz_Precalc(oldEz, Dz1, Dz2, Ca, Cb, Cc) \
  ((Ca) * (oldEz) + (Cb) * (Dz1) - (Cc) * (Dz2))

#define calculateHx_from_Bx_Precalc(oldHx, Bx1, Bx2, Da, Db, Dc) \
  ((Da) * (oldHx) + (Db) * (Bx1) - (Dc) * (Bx2))

#define calculateHy_from_By_Precalc(oldHy, By1, By2, Da, Db, Dc) \
  ((Da) * (oldHy) + (Db) * (By1) - (Dc) * (By2))

#define calculateHz_from_Bz_Precalc(oldHz, Bz1, Bz2, Da, Db, Dc) \
  ((Da) * (oldHz) + (Db) * (Bz1) - (Dc) * (Bz2))

#endif /* KERNELS_H */

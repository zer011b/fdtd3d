#ifndef KERNELS_H
#define KERNELS_H

#include "FieldValue.h"

/*
 * For all kernels index 1 represents field value with higher coordinate than index 2.
 *
 * FIXME: add docs link to formulas description
 */

/* 3D Kernels with precalculated coefficients */
#define calculateEx_3D_Precalc(oldEx, Hz1, Hz2, Hy1, Hy2, Jx_delta, Ca, Cb) \
  ((Ca) * (oldEx) + (Cb) * ((Hz1) - (Hz2) - (Hy1) + (Hy2) + Jx_delta))

#define calculateEy_3D_Precalc(oldEy, Hx1, Hx2, Hz1, Hz2, Jy_delta, Ca, Cb) \
  ((Ca) * (oldEy) + (Cb) * ((Hx1) - (Hx2) - (Hz1) + (Hz2) + Jy_delta))

#define calculateEz_3D_Precalc(oldEz, Hy1, Hy2, Hx1, Hx2, Jz_delta, Ca, Cb) \
  ((Ca) * (oldEz) + (Cb) * ((Hy1) - (Hy2) - (Hx1) + (Hx2) + Jz_delta))

#define calculateHx_3D_Precalc(oldHx, Ey1, Ey2, Ez1, Ez2, Mx_delta, Da, Db) \
  ((Da) * (oldHx) + (Db) * ((Ey1) - (Ey2) - (Ez1) + (Ez2) + Mx_delta))

#define calculateHy_3D_Precalc(oldHy, Ez1, Ez2, Ex1, Ex2, My_delta, Da, Db) \
  ((Da) * (oldHy) + (Db) * ((Ez1) - (Ez2) - (Ex1) + (Ex2) + My_delta))

#define calculateHz_3D_Precalc(oldHz, Ex1, Ex2, Ey1, Ey2, Mz_delta, Da, Db) \
  ((Da) * (oldHz) + (Db) * ((Ex1) - (Ex2) - (Ey1) + (Ey2) + Mz_delta))

/* 2D Kernels with precalculated coefficients */
#define calculateEx_2D_TEz_Precalc(oldEx, Hz1, Hz2, Ca, Cb) \
  calculateEx_3D_Precalc(oldEx, Hz1, Hz2, FPValue(0.0), FPValue(0.0), Ca, Cb)

#define calculateEy_2D_TEz_Precalc(oldEy, Hz1, Hz2, Ca, Cb) \
  calculateEy_3D_Precalc(oldEy, FPValue(0.0), FPValue(0.0), Hz1, Hz2, Ca, Cb)

#define calculateHx_2D_TMz_Precalc(oldHx, Ez1, Ez2, Da, Db) \
  calculateHx_3D_Precalc(oldHx, FPValue(0.0), FPValue(0.0), Ez1, Ez2, Da, Db)

#define calculateHy_2D_TMz_Precalc(oldHy, Ez1, Ez2, Da, Db) \
  calculateHy_3D_Precalc(oldHy, Ez1, Ez2, FPValue(0.0), FPValue(0.0), Da, Db)

/* 3D Kernels */
#define calculateEx_3D(oldEx, Hz1, Hz2, Hy1, Hy2, Jx, dt, dx, eps) \
  calculateEx_3D_Precalc(oldEx, Hz1, Hz2, Hy1, Hy2, (Jx)*(dx), FPValue(1.0), (dt) / ((eps) * (dx)))

#define calculateEy_3D(oldEy, Hx1, Hx2, Hz1, Hz2, Jy, dt, dx, eps) \
  calculateEy_3D_Precalc(oldEy, Hx1, Hx2, Hz1, Hz2, (Jy)*(dx), FPValue(1.0), (dt) / ((eps) * (dx)))

#define calculateEz_3D(oldEz, Hy1, Hy2, Hx1, Hx2, Jz, dt, dx, eps) \
  calculateEz_3D_Precalc(oldEz, Hy1, Hy2, Hx1, Hx2, (Jz)*(dx), FPValue(1.0), (dt) / ((eps) * (dx)))

#define calculateHx_3D(oldHx, Ey1, Ey2, Ez1, Ez2, Mx, dt, dx, mu) \
  calculateHx_3D_Precalc(oldHx, Ey1, Ey2, Ez1, Ez2, (Mx)*(dx), FPValue(1.0), (dt) / ((mu) * (dx)))

#define calculateHy_3D(oldHy, Ez1, Ez2, Ex1, Ex2, My, dt, dx, mu) \
  calculateHy_3D_Precalc(oldHy, Ez1, Ez2, Ex1, Ex2, (My)*(dx), FPValue(1.0), (dt) / ((mu) * (dx)))

#define calculateHz_3D(oldHz, Ex1, Ex2, Ey1, Ey2, Mz, dt, dx, mu) \
  calculateHz_3D_Precalc(oldHz, Ex1, Ex2, Ey1, Ey2, (Mz)*(dx), FPValue(1.0), (dt) / ((mu) * (dx)))

/* 2D Kernels */
#define calculateEx_2D_TEz(oldEx, Hz1, Hz2, dt, dx, eps) \
  calculateEx_3D(oldEx, Hz1, Hz2, FPValue(0.0), FPValue(0.0), dt, dx, eps)

#define calculateEy_2D_TEz(oldEy, Hz1, Hz2, dt, dx, eps) \
  calculateEy_3D(oldEy, FPValue(0.0), FPValue(0.0), Hz1, Hz2, dt, dx, eps)

#define calculateHx_2D_TMz(oldHx, Ez1, Ez2, dt, dx, mu) \
  calculateHx_3D(oldHx, FPValue(0.0), FPValue(0.0), Ez1, Ez2, dt, dx, mu)

#define calculateHy_2D_TMz(oldHy, Ez1, Ez2, dt, dx, mu) \
  calculateHy_3D(oldHy, Ez1, Ez2, FPValue(0.0), FPValue(0.0), dt, dx, mu)

/* Kernels to calculate E from D and H from B */
/*
 * FIXME: unify
 */
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



/*
 * FIXME: unsafe
 */
#define calculateDrudeE(nextD, curD, prevD, curE, prevE, b0, b1, b2, a1, a2) \
  ((b0) * (nextD) + (b1) * (curD) + (b2) * (prevD) - (a1) * (curE) - (a2) * (prevE))

#define calculateDrudeH(nextB, curB, prevB, curH, prevH, d0, d1, d2, c1, c2) \
  ((d0) * (nextB) + (d1) * (curB) + (d2) * (prevB) - (c1) * (curH) - (c2) * (prevH))

#endif /* KERNELS_H */

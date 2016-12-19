#ifndef KERNELS_H
#define KERNELS_H

#include "FieldValue.h"

/* 2D TMz */
#define calculateEx_2D_TEz(oldEz, Hz1, Hz2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Hz1) - (Hz2)))

#define calculateEy_2D_TEz(oldEz, Hz1, Hz2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * (- (Hz1) + (Hz2)))

#define calculateEz_2D_TMz(oldEz, Hx1, Hx2, Hy1, Hy2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Hx1) - (Hx2) + (Hy1) - (Hy2)))

#define calculateHx_2D_TMz(oldHx, Ez1, Ez2, dt, dx, mu) \
  ((oldHx) + ((dt) / ((mu) * (dx))) * (- (Ez1) + (Ez2)))

#define calculateHy_2D_TMz(oldHy, Ez1, Ez2, dt, dx, mu) \
  ((oldHy) + ((dt) / ((mu) * (dx))) * ((Ez1) - (Ez2)))

#define calculateHz_2D_TEz(oldEz, Ex1, Ex2, Ey1, Ey2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Ex1) - (Ex2) + (Ey1) - (Ey2)))

/* 3D */
/*
 * FIXME: incorrect
 */
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




#define calculateDx_2D_Precalc(Ca, Cb, oldDx, Hz1, Hz2) \
  ((Ca) * (oldDx) + (Cb) * ((Hz1) - (Hz2)))

#define calculateEx_2D_Precalc(Ca, Cb, Cc, oldEx, Dx1, Dx2) \
  ((Ca) * (oldEx) + (Cb) * (Dx1) - (Cc) * (Dx2))

#define calculateDy_2D_Precalc(Ca, Cb, oldDy, Hz1, Hz2) \
  ((Ca) * (oldDy) + (Cb) * (- (Hz1) + (Hz2)))

#define calculateEy_2D_Precalc(Ca, Cb, Cc, oldEy, Dy1, Dy2) \
  ((Ca) * (oldEy) + (Cb) * (Dy1) - (Cc) * (Dy2))

#define calculateDz_3D_Precalc(Ca, Cb, oldDz, Hy1, Hy2, Hx1, Hx2) \
  ((Ca) * (oldDz) + (Cb) * ((Hy1) - (Hy2) - (Hx1) + (Hx2)))

#define calculateEz_3D_Precalc(Ca, Cb, Cc, oldEz, Dz1, Dz2) \
  ((Ca) * (oldEz) + (Cb) * (Dz1) - (Cc) * (Dz2))

#define calculateBx_2D_Precalc(Da, Db, oldBx, Ez1, Ez2) \
  ((Da) * (oldBx) + (Db) * (- (Ez1) + (Ez2)))

#define calculateHx_2D_Precalc(Ca, Cb, Cc, oldHx, Bx1, Bx2) \
  ((Ca) * (oldHx) + (Cb) * (Bx1) - (Cc) * (Bx2))

#define calculateBy_2D_Precalc(Da, Db, oldBy, Ez1, Ez2) \
  ((Da) * (oldBy) + (Db) * ((Ez1) - (Ez2)))

#define calculateHy_2D_Precalc(Ca, Cb, Cc, oldHy, By1, By2) \
  calculateHx_2D_Precalc (Ca, Cb, Cc, oldHy, By1, By2)

#define calculateBz_2D_Precalc(Ca, Cb, oldHz, Ex1, Ex2, Ey1, Ey2) \
  ((Ca) * (oldHz) + (Cb) * ((Ex1) - (Ex2) - (Ey1) + (Ey2)))

#define calculateHz_2D_Precalc(Ca, Cb, Cc, oldHz, Bz1, Bz2) \
  ((Ca) * (oldHz) + (Cb) * (Bz1) - (Cc) * (Bz2))

#endif /* KERNELS_H */

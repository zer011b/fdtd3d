#ifndef KERNELS_H
#define KERNELS_H

#include "FieldValue.h"

enum CudaExitStatus
{
  CUDA_OK,
  CUDA_ERROR
};

#define calculateEz_2D_TMz(oldEz, Hx1, Hx2, Hy1, Hy2, dt, dx, eps) \
  ((oldEz) + ((dt) / ((eps) * (dx))) * ((Hx1) - (Hx2) + (Hy1) - (Hy2)))

#define calculateHx_2D_TMz(oldHx, Ez1, Ez2, dt, dx, mu) \
  ((oldHx) + ((dt) / ((mu) * (dx))) * ((Ez1) - (Ez2)))

#define calculateHy_2D_TMz(oldHy, Ez1, Ez2, dt, dx, mu) \
  calculateHx_2D_TMz ((oldHy), (Ez1), (Ez2), (dt), (dx), (mu))

#endif /* KERNELS_H */

#ifndef CALLBACK_H
#define CALLBACK_H

#include "FieldValue.h"
#include "GridCoordinate3D.h"

typedef FieldValue (*SourceCallBack) (GridCoordinateFP3D, FPValue);

#ifdef CUDA_ENABLED
#define CALLBACK(CALLBACK_NAME) \
  extern CUDA_DEVICE SourceCallBack d_ ##CALLBACK_NAME;

CALLBACK (polinom1_ez)
CALLBACK (polinom1_hy)
CALLBACK (polinom1_jz)
CALLBACK (polinom1_my)

CALLBACK (polinom2_ex)
CALLBACK (polinom2_ey)
CALLBACK (polinom2_ez)
CALLBACK (polinom2_hx)
CALLBACK (polinom2_hy)
CALLBACK (polinom2_hz)

CALLBACK (polinom2_jx)
CALLBACK (polinom2_jy)
CALLBACK (polinom2_jz)
CALLBACK (polinom2_mx)
CALLBACK (polinom2_my)
CALLBACK (polinom2_mz)

CALLBACK (polinom3_ez)
CALLBACK (polinom3_hy)
CALLBACK (polinom3_jz)
CALLBACK (polinom3_my)

CALLBACK (sin1_ez)
CALLBACK (sin1_hy)

CALLBACK (exp1_ex_exhy)
CALLBACK (exp2_ex_exhy)
CALLBACK (exp3_ex_exhy)
CALLBACK (exp1_hy_exhy)
CALLBACK (exp2_hy_exhy)
CALLBACK (exp3_hy_exhy)

CALLBACK (exp1_ex_exhz)
CALLBACK (exp2_ex_exhz)
CALLBACK (exp3_ex_exhz)
CALLBACK (exp1_hz_exhz)
CALLBACK (exp2_hz_exhz)
CALLBACK (exp3_hz_exhz)

CALLBACK (exp1_ey_eyhx)
CALLBACK (exp2_ey_eyhx)
CALLBACK (exp3_ey_eyhx)
CALLBACK (exp1_hx_eyhx)
CALLBACK (exp2_hx_eyhx)
CALLBACK (exp3_hx_eyhx)

CALLBACK (exp1_ey_eyhz)
CALLBACK (exp2_ey_eyhz)
CALLBACK (exp3_ey_eyhz)
CALLBACK (exp1_hz_eyhz)
CALLBACK (exp2_hz_eyhz)
CALLBACK (exp3_hz_eyhz)

CALLBACK (exp1_ez_ezhx)
CALLBACK (exp2_ez_ezhx)
CALLBACK (exp3_ez_ezhx)
CALLBACK (exp1_hx_ezhx)
CALLBACK (exp2_hx_ezhx)
CALLBACK (exp3_hx_ezhx)

CALLBACK (exp1_ez_ezhy)
CALLBACK (exp2_ez_ezhy)
CALLBACK (exp3_ez_ezhy)
CALLBACK (exp1_hy_ezhy)
CALLBACK (exp2_hy_ezhy)
CALLBACK (exp3_hy_ezhy)

#undef CALLBACK
#endif /* CUDA_ENABLED */

class CallBack
{
public:

#define CALLBACK(CALLBACK_NAME) \
  static CUDA_DEVICE CUDA_HOST FieldValue CALLBACK_NAME (GridCoordinateFP3D, FPValue);

  CALLBACK (polinom1_ez)
  CALLBACK (polinom1_hy)
  CALLBACK (polinom1_jz)
  CALLBACK (polinom1_my)

  CALLBACK (polinom2_ex)
  CALLBACK (polinom2_ey)
  CALLBACK (polinom2_ez)
  CALLBACK (polinom2_hx)
  CALLBACK (polinom2_hy)
  CALLBACK (polinom2_hz)

  CALLBACK (polinom2_jx)
  CALLBACK (polinom2_jy)
  CALLBACK (polinom2_jz)
  CALLBACK (polinom2_mx)
  CALLBACK (polinom2_my)
  CALLBACK (polinom2_mz)

  CALLBACK (polinom3_ez)
  CALLBACK (polinom3_hy)
  CALLBACK (polinom3_jz)
  CALLBACK (polinom3_my)

  CALLBACK (sin1_ez)
  CALLBACK (sin1_hy)

  CALLBACK (exp1_ex_exhy)
  CALLBACK (exp2_ex_exhy)
  CALLBACK (exp3_ex_exhy)
  CALLBACK (exp1_hy_exhy)
  CALLBACK (exp2_hy_exhy)
  CALLBACK (exp3_hy_exhy)

  CALLBACK (exp1_ex_exhz)
  CALLBACK (exp2_ex_exhz)
  CALLBACK (exp3_ex_exhz)
  CALLBACK (exp1_hz_exhz)
  CALLBACK (exp2_hz_exhz)
  CALLBACK (exp3_hz_exhz)

  CALLBACK (exp1_ey_eyhx)
  CALLBACK (exp2_ey_eyhx)
  CALLBACK (exp3_ey_eyhx)
  CALLBACK (exp1_hx_eyhx)
  CALLBACK (exp2_hx_eyhx)
  CALLBACK (exp3_hx_eyhx)

  CALLBACK (exp1_ey_eyhz)
  CALLBACK (exp2_ey_eyhz)
  CALLBACK (exp3_ey_eyhz)
  CALLBACK (exp1_hz_eyhz)
  CALLBACK (exp2_hz_eyhz)
  CALLBACK (exp3_hz_eyhz)

  CALLBACK (exp1_ez_ezhx)
  CALLBACK (exp2_ez_ezhx)
  CALLBACK (exp3_ez_ezhx)
  CALLBACK (exp1_hx_ezhx)
  CALLBACK (exp2_hx_ezhx)
  CALLBACK (exp3_hx_ezhx)

  CALLBACK (exp1_ez_ezhy)
  CALLBACK (exp2_ez_ezhy)
  CALLBACK (exp3_ez_ezhy)
  CALLBACK (exp1_hy_ezhy)
  CALLBACK (exp2_hy_ezhy)
  CALLBACK (exp3_hy_ezhy)

#undef CALLBACK

private:
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_e (FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_e (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_e (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_h (FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_h (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_h (FPValue, FPValue, FPValue, FPValue);
};

#endif /* CALLBACK_H */

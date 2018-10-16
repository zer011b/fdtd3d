#ifndef CALLBACK_H
#define CALLBACK_H

#include "FieldValue.h"
#include "GridCoordinate3D.h"

typedef FieldValue (*SourceCallBack) (GridCoordinateFP3D, FPValue);

class CallBack
{
public:

  static CUDA_DEVICE CUDA_HOST FieldValue polinom1_ez (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom1_hy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom1_jz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom1_my (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_ex (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_ey (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_ez (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_hx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_hy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_hz (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_jx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_jy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_jz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_mx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_my (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom2_mz (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue polinom3_ez (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom3_hy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom3_jz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue polinom3_my (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue sin1_ez (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue sin1_hy (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ex_exhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ex_exhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ex_exhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hy_exhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hy_exhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hy_exhy (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ex_exhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ex_exhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ex_exhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hz_exhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hz_exhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hz_exhz (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ey_eyhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ey_eyhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ey_eyhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hx_eyhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hx_eyhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hx_eyhx (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ey_eyhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ey_eyhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ey_eyhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hz_eyhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hz_eyhz (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hz_eyhz (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ez_ezhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ez_ezhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ez_ezhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hx_ezhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hx_ezhx (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hx_ezhx (GridCoordinateFP3D, FPValue);

  static CUDA_DEVICE CUDA_HOST FieldValue exp1_ez_ezhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_ez_ezhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_ez_ezhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_hy_ezhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_hy_ezhy (GridCoordinateFP3D, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_hy_ezhy (GridCoordinateFP3D, FPValue);

private:
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_e (FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_e (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_e (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp1_h (FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp2_h (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FieldValue exp3_h (FPValue, FPValue, FPValue, FPValue);
};

#endif /* CALLBACK_H */

#ifndef CALLBACK_H
#define CALLBACK_H

#include "FieldValue.h"
#include "GridCoordinate3D.h"

typedef FieldValue (*SourceCallBack) (GridCoordinateFP3D, FPValue);

class CallBack
{
public:

  static FieldValue polinom1_ez (GridCoordinateFP3D, FPValue);
  static FieldValue polinom1_hy (GridCoordinateFP3D, FPValue);
  static FieldValue polinom1_jz (GridCoordinateFP3D, FPValue);
  static FieldValue polinom1_my (GridCoordinateFP3D, FPValue);

  static FieldValue polinom2_ex (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_ey (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_ez (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_hx (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_hy (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_hz (GridCoordinateFP3D, FPValue);

  static FieldValue polinom2_jx (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_jy (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_jz (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_mx (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_my (GridCoordinateFP3D, FPValue);
  static FieldValue polinom2_mz (GridCoordinateFP3D, FPValue);

  static FieldValue polinom3_ez (GridCoordinateFP3D, FPValue);
  static FieldValue polinom3_hy (GridCoordinateFP3D, FPValue);
  static FieldValue polinom3_jz (GridCoordinateFP3D, FPValue);
  static FieldValue polinom3_my (GridCoordinateFP3D, FPValue);

  static FieldValue sin1_ez (GridCoordinateFP3D, FPValue);
  static FieldValue sin1_hy (GridCoordinateFP3D, FPValue);
};

#endif /* CALLBACK_H */

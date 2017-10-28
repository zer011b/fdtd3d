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
};

#endif /* CALLBACK_H */

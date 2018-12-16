#include "SchemeHelper.h"

#include "InternalScheme.h"

/**
 * Compute N for +-x0 on time step t+0.5 (i.e. E is used as is, H as is averaged for t and t+1)
 */
NPair
SchemeHelper::ntffN3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHz1 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valHz2 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

      FieldValue *valHy1 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valHy2 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHy1 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
          || valHy2 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      FieldValue *valHz1_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
      FieldValue *valHz2_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

      FieldValue *valHy1_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
      FieldValue *valHy2_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

      ASSERT (valHz1 != NULLPTR && valHz2 != NULLPTR && valHy1 != NULLPTR && valHy2 != NULLPTR
              && valHz1_prev != NULLPTR && valHz2_prev != NULLPTR && valHy1_prev != NULLPTR && valHy2_prev != NULLPTR);

      FieldValue Hz1 = (*valHz1 + *valHz1_prev) / FPValue (2);
      FieldValue Hz2 = (*valHz2 + *valHz2_prev) / FPValue (2);
      FieldValue Hy1 = (*valHy1 + *valHy1_prev) / FPValue (2);
      FieldValue Hy2 = (*valHy2 + *valHy2_prev) / FPValue (2);

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                   + (Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);

      sum_phi += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHz1 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valHz2 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

      FieldValue *valHx1 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valHx2 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHx1 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      FieldValue *valHz1_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
      FieldValue *valHz2_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

      FieldValue *valHx1_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
      FieldValue *valHx2_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

      ASSERT (valHz1 != NULLPTR && valHz2 != NULLPTR && valHx1 != NULLPTR && valHx2 != NULLPTR
              && valHz1_prev != NULLPTR && valHz2_prev != NULLPTR && valHx1_prev != NULLPTR && valHx2_prev != NULLPTR);

      FieldValue Hz1 = (*valHz1 + *valHz1_prev) / FPValue (2);
      FieldValue Hz2 = (*valHz2 + *valHz2_prev) / FPValue (2);
      FieldValue Hx1 = (*valHx1 + *valHx1_prev) / FPValue (2);
      FieldValue Hx2 = (*valHx2 + *valHx2_prev) / FPValue (2);

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Hx1 + Hx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1);

      sum_phi += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, coordY + 0.5, z0, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHy1 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valHy2 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

      FieldValue *valHx1 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valHx2 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHy1 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
          || valHy2 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
          || valHx1 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      FieldValue *valHy1_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
      FieldValue *valHy2_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

      FieldValue *valHx1_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
      FieldValue *valHx2_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

      ASSERT (valHy1 != NULLPTR && valHy2 != NULLPTR && valHx1 != NULLPTR && valHx2 != NULLPTR
              && valHy1_prev != NULLPTR && valHy2_prev != NULLPTR && valHx1_prev != NULLPTR && valHx2_prev != NULLPTR);

      FieldValue Hy1 = (*valHy1 + *valHy1_prev) / FPValue (2);
      FieldValue Hy2 = (*valHy2 + *valHy2_prev) / FPValue (2);
      FieldValue Hx1 = (*valHx1 + *valHx1_prev) / FPValue (2);
      FieldValue Hx2 = (*valHx2 + *valHx2_prev) / FPValue (2);

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += (-(Hy1 + Hy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);

      sum_phi += ((Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

      GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
      GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

      FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valEy12 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
      FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
      FieldValue *valEy22 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

      FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valEz12 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
      FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
      FieldValue *valEz22 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy12 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
          || valEy22 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos22)
          || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEy11 != NULLPTR && valEy12 != NULLPTR && valEy21 != NULLPTR && valEy22 != NULLPTR
              && valEz11 != NULLPTR && valEz12 != NULLPTR && valEz21 != NULLPTR && valEz22 != NULLPTR);

      FieldValue Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
      FieldValue Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);
      FieldValue Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
      FieldValue Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                   + (Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (x0==rightNTFF.get1 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (x0==rightNTFF.get1 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

      GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
      GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

      FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valEx12 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
      FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
      FieldValue *valEx22 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

      FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valEz12 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
      FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
      FieldValue *valEz22 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULLPTR && valEx12 != NULLPTR && valEx21 != NULLPTR && valEx22 != NULLPTR
              && valEz11 != NULLPTR && valEz12 != NULLPTR && valEz21 != NULLPTR && valEz22 != NULLPTR);

      FieldValue Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
      FieldValue Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
      FieldValue Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
      FieldValue Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Ex1 + Ex2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
      GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, coordY + 0.5, z0, ct1, ct2, ct3);

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

      GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
      GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

      FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
      FieldValue *valEx12 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
      FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
      FieldValue *valEx22 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

      FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
      FieldValue *valEy12 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
      FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
      FieldValue *valEy22 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
          || valEy12 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos32)
          || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41)
          || valEy22 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULLPTR && valEx12 != NULLPTR && valEx21 != NULLPTR && valEx22 != NULLPTR
              && valEy11 != NULLPTR && valEy12 != NULLPTR && valEy21 != NULLPTR && valEy22 != NULLPTR);

      FieldValue Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
      FieldValue Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
      FieldValue Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
      FieldValue Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ey1 + Ey2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   - (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);

      sum_phi += ((Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (anglePhi))
                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (-1) * (z0==rightNTFF.get3 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YL3D_Dim3 *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate3D> *curEz,
                       Grid<GridCoordinate3D> *curHx,
                       Grid<GridCoordinate3D> *curHy,
                       Grid<GridCoordinate3D> *curHz)
{
  NPair nx = ntffN3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHy, curHz)
                     + ntffN3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHy, curHz);
  NPair ny = ntffN3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHx, curHz)
                     + ntffN3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHx, curHz);
  NPair nz = ntffN3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHx, curHy)
                     + ntffN3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEz, curHx, curHy);

  return nx + ny + nz;
}

NPair
SchemeHelper::ntffL3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YL3D_Dim3 *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate3D> *curEx,
                       Grid<GridCoordinate3D> *curEy,
                       Grid<GridCoordinate3D> *curEz)
{
  NPair lx = ntffL3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEy, curEz)
                     + ntffL3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEy, curEz);
  NPair ly = ntffL3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEz)
                     + ntffL3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEz);
  NPair lz = ntffL3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz)
                     + ntffL3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz);

  return lx + ly + lz;
}

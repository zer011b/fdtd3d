#include "SchemeHelper.h"

#include "InternalScheme.h"

NPair
SchemeHelper::ntffN3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHz1 = curHz->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valHz2 = curHz->getFieldValueOrNullByAbsolutePos (pos21, 1);

      FieldValue *valHy1 = curHy->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valHy2 = curHy->getFieldValueOrNullByAbsolutePos (pos41, 1);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHy1 != NULL && valHy2 != NULL);

      FieldValue Hz1 = *valHz1;
      FieldValue Hz2 = *valHz2;
      FieldValue Hy1 = *valHy1;
      FieldValue Hy2 = *valHy2;

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hz2 -= yeeLayout->getHzFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy1 -= yeeLayout->getHyFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy2 -= yeeLayout->getHyFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

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
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHz1 = curHz->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valHz2 = curHz->getFieldValueOrNullByAbsolutePos (pos21, 1);

      FieldValue *valHx1 = curHx->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valHx2 = curHx->getFieldValueOrNullByAbsolutePos (pos41, 1);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hz1 = *valHz1;
      FieldValue Hz2 = *valHz2;
      FieldValue Hx1 = *valHx1;
      FieldValue Hx2 = *valHx2;

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hz2 -= yeeLayout->getHzFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx1 -= yeeLayout->getHxFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx2 -= yeeLayout->getHxFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

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
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldValue *valHy1 = curHy->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valHy2 = curHy->getFieldValueOrNullByAbsolutePos (pos21, 1);

      FieldValue *valHx1 = curHx->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valHx2 = curHx->getFieldValueOrNullByAbsolutePos (pos41, 1);

#ifdef PARALLEL_GRID
      if (valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHy1 != NULL && valHy2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hy1 = *valHy1;
      FieldValue Hy2 = *valHy2;
      FieldValue Hx1 = *valHx1;
      FieldValue Hx2 = *valHx2;

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Hy1 -= yeeLayout->getHyFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy2 -= yeeLayout->getHyFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx1 -= yeeLayout->getHxFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx2 -= yeeLayout->getHxFromIncidentH (InternalSchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

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
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      FieldValue *valEy11 = curEy->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valEy12 = curEy->getFieldValueOrNullByAbsolutePos (pos12, 1);
      FieldValue *valEy21 = curEy->getFieldValueOrNullByAbsolutePos (pos21, 1);
      FieldValue *valEy22 = curEy->getFieldValueOrNullByAbsolutePos (pos22, 1);

      FieldValue *valEz11 = curEz->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valEz12 = curEz->getFieldValueOrNullByAbsolutePos (pos32, 1);
      FieldValue *valEz21 = curEz->getFieldValueOrNullByAbsolutePos (pos41, 1);
      FieldValue *valEz22 = curEz->getFieldValueOrNullByAbsolutePos (pos42, 1);

#ifdef PARALLEL_GRID
      if (valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
      FieldValue Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);
      FieldValue Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
      FieldValue Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Ey1 -= yeeLayout->getEyFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey2 -= yeeLayout->getEyFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez1 -= yeeLayout->getEzFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez2 -= yeeLayout->getEzFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                   + (Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);
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
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      FieldValue *valEx11 = curEx->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valEx12 = curEx->getFieldValueOrNullByAbsolutePos (pos12, 1);
      FieldValue *valEx21 = curEx->getFieldValueOrNullByAbsolutePos (pos21, 1);
      FieldValue *valEx22 = curEx->getFieldValueOrNullByAbsolutePos (pos22, 1);

      FieldValue *valEz11 = curEz->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valEz12 = curEz->getFieldValueOrNullByAbsolutePos (pos32, 1);
      FieldValue *valEz21 = curEz->getFieldValueOrNullByAbsolutePos (pos41, 1);
      FieldValue *valEz22 = curEz->getFieldValueOrNullByAbsolutePos (pos42, 1);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
      FieldValue Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
      FieldValue Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
      FieldValue Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ex2 -= yeeLayout->getExFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez1 -= yeeLayout->getEzFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez2 -= yeeLayout->getEzFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Ex1 + Ex2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1);
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
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));

      FieldValue *valEx11 = curEx->getFieldValueOrNullByAbsolutePos (pos11, 1);
      FieldValue *valEx12 = curEx->getFieldValueOrNullByAbsolutePos (pos12, 1);
      FieldValue *valEx21 = curEx->getFieldValueOrNullByAbsolutePos (pos21, 1);
      FieldValue *valEx22 = curEx->getFieldValueOrNullByAbsolutePos (pos22, 1);

      FieldValue *valEy11 = curEy->getFieldValueOrNullByAbsolutePos (pos31, 1);
      FieldValue *valEy12 = curEy->getFieldValueOrNullByAbsolutePos (pos32, 1);
      FieldValue *valEy21 = curEy->getFieldValueOrNullByAbsolutePos (pos41, 1);
      FieldValue *valEy22 = curEy->getFieldValueOrNullByAbsolutePos (pos42, 1);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos32)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL);

      FieldValue Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
      FieldValue Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
      FieldValue Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
      FieldValue Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);

      if (SOLVER_SETTINGS.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ex2 -= yeeLayout->getExFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey1 -= yeeLayout->getEyFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey2 -= yeeLayout->getEyFromIncidentE (InternalSchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += (-(Ey1 + Ey2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);

      sum_phi += ((Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (anglePhi))
                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);
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
                       Grid<GridCoordinate1D> *HInc,
                       Grid<GridCoordinate3D> *curEz,
                       Grid<GridCoordinate3D> *curHx,
                       Grid<GridCoordinate3D> *curHy,
                       Grid<GridCoordinate3D> *curHz)
{
  NPair nx = SchemeHelper::ntffN3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHy, curHz)
                     + ntffN3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHy, curHz);
  NPair ny = SchemeHelper::ntffN3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHz)
                     + SchemeHelper::ntffN3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHz);
  NPair nz = SchemeHelper::ntffN3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy)
                     + SchemeHelper::ntffN3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy);

  return nx + ny + nz;
}

NPair
SchemeHelper::ntffL3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YL3D_Dim3 *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate1D> *EInc,
                       Grid<GridCoordinate3D> *curEx,
                       Grid<GridCoordinate3D> *curEy,
                       Grid<GridCoordinate3D> *curEz)
{
  NPair lx = SchemeHelper::ntffL3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEy, curEz)
                     + SchemeHelper::ntffL3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEy, curEz);
  NPair ly = SchemeHelper::ntffL3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEz)
                     + SchemeHelper::ntffL3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEz);
  NPair lz = SchemeHelper::ntffL3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz)
                     + SchemeHelper::ntffL3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz);

  return lx + ly + lz;
}

/*
 * Unit test for basic operations with Grid
 */

#include <iostream>

#include "Assert.h"
#include "GridCoordinate3D.h"
#include "Grid.h"

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

const FPValue imagMult = 1000;
const FPValue prevMult = 16;
const FPValue prevPrevMult = prevMult * prevMult;

const int gridSizeX = 32;
const int gridSizeY = 32;
const int gridSizeZ = 32;

static FieldValue updateVal (FPValue fpval, int time_step_back)
{
  if (time_step_back == 0)
  {
#ifdef COMPLEX_FIELD_VALUES
    return FieldValue (fpval, fpval * imagMult);
#else /* COMPLEX_FIELD_VALUES */  
    return FieldValue (fpval);
#endif /* !COMPLEX_FIELD_VALUES */
  }
  
  if (time_step_back == 1)
  {
#ifdef COMPLEX_FIELD_VALUES
    return FieldValue (fpval * prevMult, fpval * prevMult * imagMult);
#else /* COMPLEX_FIELD_VALUES */  
    return FieldValue (fpval * prevMult);
#endif /* !COMPLEX_FIELD_VALUES */
  }
  
  if (time_step_back == 2)
  {
#ifdef COMPLEX_FIELD_VALUES
    return FieldValue (fpval * prevPrevMult, fpval * prevPrevMult * imagMult);
#else /* COMPLEX_FIELD_VALUES */  
    return FieldValue (fpval * prevPrevMult);
#endif /* !COMPLEX_FIELD_VALUES */
  }

  UNREACHABLE;
}

static void checkIsTheSame (Grid<GridCoordinate1D> *grid1D,
                            Grid<GridCoordinate2D> *grid2D,
                            Grid<GridCoordinate3D> *grid3D)
{
  for (grid_coord i = 0; i < gridSizeX; ++i)
  {
    for (grid_coord j = 0; j < gridSizeY; ++j)
    {
      for (grid_coord k = 0; k < gridSizeZ; ++k)
      {
        GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k,
                                                   grid3D->getSize ().getType1 (),
                                                   grid3D->getSize ().getType2 (),
                                                   grid3D->getSize ().getType3 ());
        grid_coord coord = grid3D->calculateIndexFromPosition (pos);

        FPValue fpval = i * j * k;
        for (int t = 0; t < 3; ++t)
        {
          FieldValue val_old = updateVal (fpval, t);
          ASSERT (val_old == *grid3D->getFieldValue (coord, t));
        }
      }
      
      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j,
                                                 grid2D->getSize ().getType1 (),
                                                 grid2D->getSize ().getType2 ());
      grid_coord coord = grid2D->calculateIndexFromPosition (pos);

      FPValue fpval = i * j;
      for (int t = 0; t < 3; ++t)
      {
        FieldValue val_old = updateVal (fpval, t);
        ASSERT (val_old == *grid2D->getFieldValue (coord, t));
      }
    }
    
    GridCoordinate1D pos = GRID_COORDINATE_1D (i,
                                               grid1D->getSize ().getType1 ());
    grid_coord coord = grid1D->calculateIndexFromPosition (pos);

    FPValue fpval = i;
    for (int t = 0; t < 3; ++t)
    {
      FieldValue val_old = updateVal (fpval, t);
      ASSERT (val_old == *grid1D->getFieldValue (coord, t));
    }
  }
}

static void bmp (Grid<GridCoordinate1D> *grid1D,
                 Grid<GridCoordinate2D> *grid2D,
                 Grid<GridCoordinate3D> *grid3D)
{
  BMPDumper<GridCoordinate1D> bmpDumper1D;
  bmpDumper1D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);
  BMPDumper<GridCoordinate2D> bmpDumper2D;
  bmpDumper2D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);
  BMPDumper<GridCoordinate3D> bmpDumper3D;
  bmpDumper3D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);

  BMPLoader<GridCoordinate1D> bmpLoader1D;
  bmpLoader1D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);
  BMPLoader<GridCoordinate2D> bmpLoader2D;
  bmpLoader2D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);
  BMPLoader<GridCoordinate3D> bmpLoader3D;
  bmpLoader3D.initializeHelper (PaletteType::PALETTE_GRAY, OrthogonalAxis::Z);

  GridCoordinate1D pos1D = GRID_COORDINATE_1D (0, grid1D->getSize ().getType1 ());
  bmpDumper1D.dumpGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);

  GridCoordinate2D pos2D = GRID_COORDINATE_2D (0, 0, grid2D->getSize ().getType1 (), grid2D->getSize ().getType2 ());
  bmpDumper2D.dumpGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);

  GridCoordinate3D pos3D = GRID_COORDINATE_3D (0, 0, 0,
                                               grid3D->getSize ().getType1 (),
                                               grid3D->getSize ().getType2 (),
                                               grid3D->getSize ().getType3 ());
  bmpDumper3D.dumpGrid (grid3D, pos3D, grid3D->getSize (), 0, -1);

  bmpLoader1D.loadGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);
  bmpLoader2D.loadGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);
  // bmpLoader3D.loadGrid (grid3D, pos3D, grid3D->getSize (), 0, -1); /* UNIMPLEMENTED */
}

static void dat (Grid<GridCoordinate1D> *grid1D,
                 Grid<GridCoordinate2D> *grid2D,
                 Grid<GridCoordinate3D> *grid3D)
{
  DATDumper<GridCoordinate1D> datDumper1D;
  DATDumper<GridCoordinate2D> datDumper2D;
  DATDumper<GridCoordinate3D> datDumper3D;

  DATLoader<GridCoordinate1D> datLoader1D;
  DATLoader<GridCoordinate2D> datLoader2D;
  DATLoader<GridCoordinate3D> datLoader3D;

  GridCoordinate1D pos1D = GRID_COORDINATE_1D (0, grid1D->getSize ().getType1 ());
  datDumper1D.dumpGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);

  GridCoordinate2D pos2D = GRID_COORDINATE_2D (0, 0, grid2D->getSize ().getType1 (), grid2D->getSize ().getType2 ());
  datDumper2D.dumpGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);

  GridCoordinate3D pos3D = GRID_COORDINATE_3D (0, 0, 0,
                                               grid3D->getSize ().getType1 (),
                                               grid3D->getSize ().getType2 (),
                                               grid3D->getSize ().getType3 ());
  datDumper3D.dumpGrid (grid3D, pos3D, grid3D->getSize (), 0, -1);

  datLoader1D.loadGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);
  datLoader2D.loadGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);
  datLoader3D.loadGrid (grid3D, pos3D, grid3D->getSize (), 0, -1);

  checkIsTheSame (grid1D, grid2D, grid3D);
}

static void txt (Grid<GridCoordinate1D> *grid1D,
                 Grid<GridCoordinate2D> *grid2D,
                 Grid<GridCoordinate3D> *grid3D)
{
  TXTDumper<GridCoordinate1D> txtDumper1D;
  TXTDumper<GridCoordinate2D> txtDumper2D;
  TXTDumper<GridCoordinate3D> txtDumper3D;

  TXTLoader<GridCoordinate1D> txtLoader1D;
  TXTLoader<GridCoordinate2D> txtLoader2D;
  TXTLoader<GridCoordinate3D> txtLoader3D;

  GridCoordinate1D pos1D = GRID_COORDINATE_1D (0, grid1D->getSize ().getType1 ());
  txtDumper1D.dumpGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);

  GridCoordinate2D pos2D = GRID_COORDINATE_2D (0, 0, grid2D->getSize ().getType1 (), grid2D->getSize ().getType2 ());
  txtDumper2D.dumpGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);

  GridCoordinate3D pos3D = GRID_COORDINATE_3D (0, 0, 0,
                                               grid3D->getSize ().getType1 (),
                                               grid3D->getSize ().getType2 (),
                                               grid3D->getSize ().getType3 ());
  txtDumper3D.dumpGrid (grid3D, pos3D, grid3D->getSize (), 0, -1);

  txtLoader1D.loadGrid (grid1D, pos1D, grid1D->getSize (), 0, -1);
  txtLoader2D.loadGrid (grid2D, pos2D, grid2D->getSize (), 0, -1);
  txtLoader3D.loadGrid (grid3D, pos3D, grid3D->getSize (), 0, -1);

  checkIsTheSame (grid1D, grid2D, grid3D);
}

int main (int argc, char** argv)
{
  GridCoordinate1D overallSize1D (gridSizeX
#ifdef DEBUG_INFO
                                  , CoordinateType::X
#endif
                                  );
  GridCoordinate2D overallSize2D (gridSizeX, gridSizeY
#ifdef DEBUG_INFO
                                  , CoordinateType::X
                                  , CoordinateType::Y
#endif
                                  );
  GridCoordinate3D overallSize3D (gridSizeX, gridSizeY, gridSizeZ
#ifdef DEBUG_INFO
                                  , CoordinateType::X
                                  , CoordinateType::Y
                                  , CoordinateType::Z
#endif
                                  );

  Grid<GridCoordinate1D> grid1D (overallSize1D, 0, 3, "1D");
  Grid<GridCoordinate2D> grid2D (overallSize2D, 0, 3, "2D");
  Grid<GridCoordinate3D> grid3D (overallSize3D, 0, 3, "3D");

  for (grid_coord i = 0; i < gridSizeX; ++i)
  {
    for (grid_coord j = 0; j < gridSizeY; ++j)
    {
      for (grid_coord k = 0; k < gridSizeZ; ++k)
      {
        GridCoordinate3D pos = GRID_COORDINATE_3D (i, j, k,
                                                   grid3D.getSize ().getType1 (),
                                                   grid3D.getSize ().getType2 (),
                                                   grid3D.getSize ().getType3 ());
        grid_coord coord = grid3D.calculateIndexFromPosition (pos);

        FPValue fpval = i * j * k;
        for (int t = 0; t < 3; ++t)
        {
          grid3D.setFieldValue (updateVal (fpval, t), coord, t);
        }
      }

      GridCoordinate2D pos = GRID_COORDINATE_2D (i, j,
                                                 grid2D.getSize ().getType1 (),
                                                 grid2D.getSize ().getType2 ());
      grid_coord coord = grid2D.calculateIndexFromPosition (pos);

      FPValue fpval = i * j;
      for (int t = 0; t < 3; ++t)
      {
        grid2D.setFieldValue (updateVal (fpval, t), coord, t);
      }
    }

    GridCoordinate1D pos = GRID_COORDINATE_1D (i,
                                               grid1D.getSize ().getType1 ());
    grid_coord coord = grid1D.calculateIndexFromPosition (pos);

    FPValue fpval = i;
    for (int t = 0; t < 3; ++t)
    {
      grid1D.setFieldValue (updateVal (fpval, t), coord, t);
    }
  }

  dat (&grid1D, &grid2D, &grid3D);
  txt (&grid1D, &grid2D, &grid3D);
  bmp (&grid1D, &grid2D, &grid3D);

  return 0;
} /* main */

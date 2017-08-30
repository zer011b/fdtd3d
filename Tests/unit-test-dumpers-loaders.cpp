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

static void updateVal (FieldPointValue *val, FPValue fpval)
{
#ifdef COMPLEX_FIELD_VALUES

  val->setCurValue (FieldValue (fpval, fpval * imagMult));

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  val->setPrevValue (FieldValue (fpval * prevMult, fpval * prevMult * imagMult));
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
  val->setPrevPrevValue (FieldValue (fpval * prevPrevMult, fpval * prevPrevMult * imagMult));
#endif /* TWO_TIME_STEPS */

#else /* COMPLEX_FIELD_VALUES */

  val->setCurValue (fpval);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  val->setPrevValue (fpval * prevMult);
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
  val->setPrevPrevValue (fpval * prevPrevMult);
#endif /* TWO_TIME_STEPS */

#endif /* !COMPLEX_FIELD_VALUES */
}

static void checkIsTheSame (Grid<GridCoordinate1D> *grid1D,
                            Grid<GridCoordinate2D> *grid2D,
                            Grid<GridCoordinate3D> *grid3D)
{
  for (grid_iter i = 0; i < gridSizeX; ++i)
  {
    for (grid_iter j = 0; j < gridSizeY; ++j)
    {
      for (grid_iter k = 0; k < gridSizeZ; ++k)
      {
        FieldPointValue val_old;
        GridCoordinate3D pos (i, j, k);

        FPValue fpval = i * j * k;
        updateVal (&val_old, fpval);

        FieldPointValue *val = grid3D->getFieldPointValue (pos);

        ASSERT (*val == val_old);
      }

      FieldPointValue val_old;
      GridCoordinate2D pos (i, j);

      FPValue fpval = i * j;
      updateVal (&val_old, fpval);

      FieldPointValue *val = grid2D->getFieldPointValue (pos);

      ASSERT (*val == val_old);
    }

    FieldPointValue val_old;
    GridCoordinate1D pos (i);

    FPValue fpval = i;
    updateVal (&val_old, fpval);

    FieldPointValue *val = grid1D->getFieldPointValue (pos);

    ASSERT (*val == val_old);
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

  bmpDumper1D.init (0, ALL, 0, "tmp1D");
  bmpLoader1D.init (0, ALL, 0, "tmp1D");
  bmpDumper1D.dumpGrid (grid1D, GridCoordinate1D (0), grid1D->getSize ());

  bmpDumper2D.init (0, ALL, 0, "tmp2D");
  bmpLoader2D.init (0, ALL, 0, "tmp2D");
  bmpDumper2D.dumpGrid (grid2D, GridCoordinate2D (0, 0), grid2D->getSize ());

  bmpDumper3D.init (0, ALL, 0, "tmp3D");
  bmpLoader3D.init (0, ALL, 0, "tmp3D");
  bmpDumper3D.dumpGrid (grid3D, GridCoordinate3D (0, 0, 0), grid3D->getSize ());

  bmpLoader1D.loadGrid (grid1D);
  bmpLoader2D.loadGrid (grid2D);
  // bmpLoader3D.loadGrid (grid3D); /* UNIMPLEMENTED */
}

static void dat (Grid<GridCoordinate1D> *grid1D,
                 Grid<GridCoordinate2D> *grid2D,
                 Grid<GridCoordinate3D> *grid3D)
{
  DATDumper<GridCoordinate1D> datDumper1D;
  datDumper1D.init (0, ALL, 0, "tmp1D");
  DATDumper<GridCoordinate2D> datDumper2D;
  datDumper2D.init (0, ALL, 0, "tmp2D");
  DATDumper<GridCoordinate3D> datDumper3D;
  datDumper3D.init (0, ALL, 0, "tmp3D");

  DATLoader<GridCoordinate1D> datLoader1D;
  datLoader1D.init (0, ALL, 0, "tmp1D");
  DATLoader<GridCoordinate2D> datLoader2D;
  datLoader2D.init (0, ALL, 0, "tmp2D");
  DATLoader<GridCoordinate3D> datLoader3D;
  datLoader3D.init (0, ALL, 0, "tmp3D");

  datDumper1D.dumpGrid (grid1D, GridCoordinate1D (0), grid1D->getSize ());
  datDumper2D.dumpGrid (grid2D, GridCoordinate2D (0, 0), grid2D->getSize ());
  datDumper3D.dumpGrid (grid3D, GridCoordinate3D (0, 0, 0), grid3D->getSize ());

  datLoader1D.loadGrid (grid1D);
  datLoader2D.loadGrid (grid2D);
  datLoader3D.loadGrid (grid3D);

  checkIsTheSame (grid1D, grid2D, grid3D);
}

static void txt (Grid<GridCoordinate1D> *grid1D,
                 Grid<GridCoordinate2D> *grid2D,
                 Grid<GridCoordinate3D> *grid3D)
{
  TXTDumper<GridCoordinate1D> txtDumper1D;
  txtDumper1D.init (0, ALL, 0, "tmp1D");
  TXTDumper<GridCoordinate2D> txtDumper2D;
  txtDumper2D.init (0, ALL, 0, "tmp2D");
  TXTDumper<GridCoordinate3D> txtDumper3D;
  txtDumper3D.init (0, ALL, 0, "tmp3D");

  TXTLoader<GridCoordinate1D> txtLoader1D;
  txtLoader1D.init (0, ALL, 0, "tmp1D");
  TXTLoader<GridCoordinate2D> txtLoader2D;
  txtLoader2D.init (0, ALL, 0, "tmp2D");
  TXTLoader<GridCoordinate3D> txtLoader3D;
  txtLoader3D.init (0, ALL, 0, "tmp3D");

  txtDumper1D.dumpGrid (grid1D, GridCoordinate1D (0), grid1D->getSize ());
  txtDumper2D.dumpGrid (grid2D, GridCoordinate2D (0, 0), grid2D->getSize ());
  txtDumper3D.dumpGrid (grid3D, GridCoordinate3D (0, 0, 0), grid3D->getSize ());

  txtLoader1D.loadGrid (grid1D);
  txtLoader2D.loadGrid (grid2D);
  txtLoader3D.loadGrid (grid3D);

  checkIsTheSame (grid1D, grid2D, grid3D);
}

int main (int argc, char** argv)
{
  GridCoordinate1D overallSize1D (gridSizeX);
  GridCoordinate2D overallSize2D (gridSizeX, gridSizeY);
  GridCoordinate3D overallSize3D (gridSizeX, gridSizeY, gridSizeZ);

  Grid<GridCoordinate1D> grid1D (overallSize1D, 0);
  Grid<GridCoordinate2D> grid2D (overallSize2D, 0);
  Grid<GridCoordinate3D> grid3D (overallSize3D, 0);

  for (grid_iter i = 0; i < gridSizeX; ++i)
  {
    for (grid_iter j = 0; j < gridSizeY; ++j)
    {
      for (grid_iter k = 0; k < gridSizeZ; ++k)
      {
        FieldPointValue* val = new FieldPointValue ();
        GridCoordinate3D pos (i, j, k);

        FPValue fpval = i * j * k;
        updateVal (val, fpval);

        grid3D.setFieldPointValue (val, pos);
      }

      FieldPointValue* val = new FieldPointValue ();
      GridCoordinate2D pos (i, j);

      FPValue fpval = i * j;
      updateVal (val, fpval);

      grid2D.setFieldPointValue (val, pos);
    }

    FieldPointValue* val = new FieldPointValue ();
    GridCoordinate1D pos (i);

    FPValue fpval = i;
    updateVal (val, fpval);

    grid1D.setFieldPointValue (val, pos);
  }

  dat (&grid1D, &grid2D, &grid3D);
  txt (&grid1D, &grid2D, &grid3D);
  bmp (&grid1D, &grid2D, &grid3D);

  return 0;
} /* main */

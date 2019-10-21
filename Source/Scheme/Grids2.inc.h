/*
 * Name, Base grid name, Number of time steps stored, Sub-time step i (t+i/2)
 */

/*
 * NOTE: time offsets for all the grids should be in sync with share process!
 */

/**
 * Material grids
 */
GRID_NAME_NO_CHECK(Eps, Eps, 1, 0)
GRID_NAME_NO_CHECK(Mu, Eps, 1, 0)

/**
 * Field grids
 */
GRID_NAME(Ex, Ex, storedSteps, 1)
GRID_NAME(Ey, Ey, storedSteps, 1)
GRID_NAME(Ez, Ez, storedSteps, 1)
GRID_NAME(Hx, Hx, storedSteps, 0)
GRID_NAME(Hy, Hy, storedSteps, 0)
GRID_NAME(Hz, Hz, storedSteps, 0)

if (SOLVER_SETTINGS.getDoUseCaCbGrids ())
{
  /**
   * Helper grids, which store precomputed coefficients
   */
  GRID_NAME(CaEx, Ex, 1, 1)
  GRID_NAME(CbEx, Ex, 1, 1)
  GRID_NAME(CaEy, Ey, 1, 1)
  GRID_NAME(CbEy, Ey, 1, 1)
  GRID_NAME(CaEz, Ez, 1, 1)
  GRID_NAME(CbEz, Ez, 1, 1)
  GRID_NAME(DaHx, Hx, 1, 0)
  GRID_NAME(DbHx, Hx, 1, 0)
  GRID_NAME(DaHy, Hy, 1, 0)
  GRID_NAME(DbHy, Hy, 1, 0)
  GRID_NAME(DaHz, Hz, 1, 0)
  GRID_NAME(DbHz, Hz, 1, 0)
}

if (SOLVER_SETTINGS.getDoUsePML ())
{
  /**
   * Helper grids used for PML
   */
  GRID_NAME(Dx, Ex, storedSteps, 1)
  GRID_NAME(Dy, Ey, storedSteps, 1)
  GRID_NAME(Dz, Ez, storedSteps, 1)
  GRID_NAME(Bx, Hx, storedSteps, 0)
  GRID_NAME(By, Hy, storedSteps, 0)
  GRID_NAME(Bz, Hz, storedSteps, 0)

  /**
   * Sigmas
   */
  GRID_NAME(SigmaX, SigmaX, 1, 0)
  GRID_NAME(SigmaY, SigmaY, 1, 0)
  GRID_NAME(SigmaZ, SigmaZ, 1, 0)

  if (SOLVER_SETTINGS.getDoUseCaCbPMLGrids ())
  {
    /**
     * Helper grids for PML
     */
    GRID_NAME(CaPMLEx, Ex, 1, 1)
    GRID_NAME(CbPMLEx, Ex, 1, 1)
    GRID_NAME(CcPMLEx, Ex, 1, 1)

    GRID_NAME(CaPMLEy, Ey, 1, 1)
    GRID_NAME(CbPMLEy, Ey, 1, 1)
    GRID_NAME(CcPMLEy, Ey, 1, 1)

    GRID_NAME(CaPMLEz, Ez, 1, 1)
    GRID_NAME(CbPMLEz, Ez, 1, 1)
    GRID_NAME(CcPMLEz, Ez, 1, 1)

    GRID_NAME(DaPMLHx, Hx, 1, 0)
    GRID_NAME(DbPMLHx, Hx, 1, 0)
    GRID_NAME(DcPMLHx, Hx, 1, 0)

    GRID_NAME(DaPMLHy, Hy, 1, 0)
    GRID_NAME(DbPMLHy, Hy, 1, 0)
    GRID_NAME(DcPMLHy, Hy, 1, 0)

    GRID_NAME(DaPMLHz, Hz, 1, 0)
    GRID_NAME(DbPMLHz, Hz, 1, 0)
    GRID_NAME(DcPMLHz, Hz, 1, 0)
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    /**
     * Auxiliary field grids used for metamaterials with PML
     */
    GRID_NAME(D1x, Ex, storedSteps, 1)
    GRID_NAME(D1y, Ey, storedSteps, 1)
    GRID_NAME(D1z, Ez, storedSteps, 1)
    GRID_NAME(B1x, Hx, storedSteps, 0)
    GRID_NAME(B1y, Hy, storedSteps, 0)
    GRID_NAME(B1z, Hz, storedSteps, 0)

    if (SOLVER_SETTINGS.getDoUseCaCbPMLMetaGrids ())
    {
      /**
       * Helper grids for metamaterials and PML
       */
      GRID_NAME(CB0Ex, Ex, 1, 1)
      GRID_NAME(CB1Ex, Ex, 1, 1)
      GRID_NAME(CB2Ex, Ex, 1, 1)
      GRID_NAME(CA1Ex, Ex, 1, 1)
      GRID_NAME(CA2Ex, Ex, 1, 1)

      GRID_NAME(CB0Ey, Ey, 1, 1)
      GRID_NAME(CB1Ey, Ey, 1, 1)
      GRID_NAME(CB2Ey, Ey, 1, 1)
      GRID_NAME(CA1Ey, Ey, 1, 1)
      GRID_NAME(CA2Ey, Ey, 1, 1)

      GRID_NAME(CB0Ez, Ez, 1, 1)
      GRID_NAME(CB1Ez, Ez, 1, 1)
      GRID_NAME(CB2Ez, Ez, 1, 1)
      GRID_NAME(CA1Ez, Ez, 1, 1)
      GRID_NAME(CA2Ez, Ez, 1, 1)

      GRID_NAME(DB0Hx, Hx, 1, 0)
      GRID_NAME(DB1Hx, Hx, 1, 0)
      GRID_NAME(DB2Hx, Hx, 1, 0)
      GRID_NAME(DA1Hx, Hx, 1, 0)
      GRID_NAME(DA2Hx, Hx, 1, 0)

      GRID_NAME(DB0Hy, Hy, 1, 0)
      GRID_NAME(DB1Hy, Hy, 1, 0)
      GRID_NAME(DB2Hy, Hy, 1, 0)
      GRID_NAME(DA1Hy, Hy, 1, 0)
      GRID_NAME(DA2Hy, Hy, 1, 0)

      GRID_NAME(DB0Hz, Hz, 1, 0)
      GRID_NAME(DB1Hz, Hz, 1, 0)
      GRID_NAME(DB2Hz, Hz, 1, 0)
      GRID_NAME(DA1Hz, Hz, 1, 0)
      GRID_NAME(DA2Hz, Hz, 1, 0)
    }
  }
}

if (SOLVER_SETTINGS.getDoUseMetamaterials ())
{
  /**
   * Metamaterial grids
   */
  GRID_NAME_NO_CHECK(OmegaPE, Eps, 1, 0)
  GRID_NAME_NO_CHECK(GammaE, Eps, 1, 0)
  GRID_NAME_NO_CHECK(OmegaPM, Eps, 1, 0)
  GRID_NAME_NO_CHECK(GammaM, Eps, 1, 0)
}

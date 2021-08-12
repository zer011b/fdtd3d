/*
 * Copyright (C) 2018 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * Material grids
 */
GRID_NAME(Eps)
GRID_NAME(Mu)

/**
 * Field grids
 */
GRID_NAME(Ex)
GRID_NAME(Ey)
GRID_NAME(Ez)
GRID_NAME(Hx)
GRID_NAME(Hy)
GRID_NAME(Hz)

/**
 * Helper grids, which store precomputed coefficients
 */
GRID_NAME(CaEx)
GRID_NAME(CbEx)
GRID_NAME(CaEy)
GRID_NAME(CbEy)
GRID_NAME(CaEz)
GRID_NAME(CbEz)
GRID_NAME(DaHx)
GRID_NAME(DbHx)
GRID_NAME(DaHy)
GRID_NAME(DbHy)
GRID_NAME(DaHz)
GRID_NAME(DbHz)

/**
 * Helper grids used for PML
 */
GRID_NAME(Dx)
GRID_NAME(Dy)
GRID_NAME(Dz)
GRID_NAME(Bx)
GRID_NAME(By)
GRID_NAME(Bz)

/**
 * Sigmas
 */
GRID_NAME(SigmaX)
GRID_NAME(SigmaY)
GRID_NAME(SigmaZ)

/**
 * Helper grids for PML
 */
GRID_NAME(CaPMLEx)
GRID_NAME(CbPMLEx)
GRID_NAME(CcPMLEx)

GRID_NAME(CaPMLEy)
GRID_NAME(CbPMLEy)
GRID_NAME(CcPMLEy)

GRID_NAME(CaPMLEz)
GRID_NAME(CbPMLEz)
GRID_NAME(CcPMLEz)

GRID_NAME(DaPMLHx)
GRID_NAME(DbPMLHx)
GRID_NAME(DcPMLHx)

GRID_NAME(DaPMLHy)
GRID_NAME(DbPMLHy)
GRID_NAME(DcPMLHy)

GRID_NAME(DaPMLHz)
GRID_NAME(DbPMLHz)
GRID_NAME(DcPMLHz)

/**
 * Auxiliary field grids used for metamaterials with PML
 */
GRID_NAME(D1x)
GRID_NAME(D1y)
GRID_NAME(D1z)
GRID_NAME(B1x)
GRID_NAME(B1y)
GRID_NAME(B1z)

/**
 * Helper grids for metamaterials and PML
 */
GRID_NAME(CB0Ex)
GRID_NAME(CB1Ex)
GRID_NAME(CB2Ex)
GRID_NAME(CA1Ex)
GRID_NAME(CA2Ex)

GRID_NAME(CB0Ey)
GRID_NAME(CB1Ey)
GRID_NAME(CB2Ey)
GRID_NAME(CA1Ey)
GRID_NAME(CA2Ey)

GRID_NAME(CB0Ez)
GRID_NAME(CB1Ez)
GRID_NAME(CB2Ez)
GRID_NAME(CA1Ez)
GRID_NAME(CA2Ez)

GRID_NAME(DB0Hx)
GRID_NAME(DB1Hx)
GRID_NAME(DB2Hx)
GRID_NAME(DA1Hx)
GRID_NAME(DA2Hx)

GRID_NAME(DB0Hy)
GRID_NAME(DB1Hy)
GRID_NAME(DB2Hy)
GRID_NAME(DA1Hy)
GRID_NAME(DA2Hy)

GRID_NAME(DB0Hz)
GRID_NAME(DB1Hz)
GRID_NAME(DB2Hz)
GRID_NAME(DA1Hz)
GRID_NAME(DA2Hz)

/**
 * Metamaterial grids
 */
GRID_NAME(OmegaPE)
GRID_NAME(GammaE)
GRID_NAME(OmegaPM)
GRID_NAME(GammaM)

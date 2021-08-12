/*
 * Copyright (C) 2017 Gleb Balykov
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char **argv)
{
  if (argc != 6)
  {
    return 1;
  }

  double lambda = atof (argv[1]);
  double dx = atof (argv[2]);
  double i = atof (argv[3]);
  double n = atof (argv[4]);
  double S = atof (argv[5]);

  double c = 299792458.0;
  double dt = dx * S / c;
  double f = c / lambda;
  double omega = 2 * M_PI * f;
  double k = 2 * M_PI / lambda;

  double arg = omega * n * dt - k * i * dx;
  double re = sin (arg);
  double im = cos (arg);
  double mod = sqrt (re * re + im * im);

  printf ("%.20f %.20f %.20f\n", re, im, mod);

  return 0;
}

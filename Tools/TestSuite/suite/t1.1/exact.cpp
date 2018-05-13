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

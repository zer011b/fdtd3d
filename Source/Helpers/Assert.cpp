#include "Assert.h"

#include <cstdlib>
#include <execinfo.h>

void program_fail ()
{
  const unsigned bufsize = 256;
  int nptrs;
  void *buffer[bufsize];
  char **strings;

  nptrs = backtrace (buffer, bufsize);
  printf ("backtrace () returned %d addresses\n", nptrs);

  strings = backtrace_symbols (buffer, nptrs);

  if (strings == NULL)
  {
    printf ("backtrace_symbols error");
    exit (1);
  }

  for (int j = 0; j < nptrs; j++)
  {
    printf ("%s\n", strings[j]);
  }

  free(strings);

  exit (1);
}

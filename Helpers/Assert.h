#ifndef ASSERT_H
#define ASSERT_H

#include <cstdlib>
#include <cstdio>

void program_fail ();

#define UNREACHABLE \
{ \
  program_fail (); \
  printf ("Unreachable executed.\n"); \
  exit (1); \
}

#define ASSERT_MESSAGE (x) \
{ \
  program_fail (); \
  printf ("Assert: %s.", x); \
  exit (1); \
}

#define ASSERT (x) \
{ \
  if (!x) \
  { \
    program_fail (); \
    printf ("Assert."); \
    exit (1); \
  } \
}

#endif /* ASSERT_H */

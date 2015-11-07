#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

void program_fail ();

#define UNREACHABLE \
{ \
  printf ("Unreachable executed.\n"); \
  program_fail (); \
}

#define ASSERT_MESSAGE (x) \
{ \
  printf ("Assert: %s.", x); \
  program_fail (); \
}

#define ASSERT (x) \
{ \
  if (!x) \
  { \
    printf ("Assert."); \
    program_fail (); \
  } \
}

#endif /* ASSERT_H */

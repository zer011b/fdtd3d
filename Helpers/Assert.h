#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

// This function is used to exit and debugging purposes.
void program_fail ();

// Indicates program point, which should not be reached.
#define UNREACHABLE \
{ \
  printf ("Unreachable executed at %s:%d.\n", __FILE__, __LINE__); \
  program_fail (); \
}

// Unconditional assert with message.
#define ASSERT_MESSAGE(x) \
{ \
  printf ("Assert '%s' at %s:%d", x, __FILE__, __LINE__); \
  program_fail (); \
}

// Conditional assert with default message.
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    printf ("Assert at %s:%d", __FILE__, __LINE__); \
    program_fail (); \
  } \
}

#endif /* ASSERT_H */

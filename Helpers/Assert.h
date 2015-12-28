#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

// This function is used to exit and debugging purposes.
void program_fail ();

// Indicates program point, which should not be reached.
#if PRINT_MESSAGE
#define UNREACHABLE \
{ \
  printf ("Unreachable executed at %s:%d.\n", __FILE__, __LINE__); \
  program_fail (); \
}
#else
#define UNREACHABLE \
{ \
  program_fail (); \
}
#endif

// Unconditional assert with message.
#if PRINT_MESSAGE
#define ASSERT_MESSAGE(x) \
{ \
  printf ("Assert '%s' at %s:%d.\n", x, __FILE__, __LINE__); \
  program_fail (); \
}
#else
#define ASSERT_MESSAGE(x) \
{ \
  program_fail (); \
}
#endif

// Conditional assert with default message.
#if PRINT_MESSAGE
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    printf ("Assert at %s:%d.\n", __FILE__, __LINE__); \
    program_fail (); \
  } \
}
#else
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    program_fail (); \
  } \
}
#endif

#endif /* ASSERT_H */

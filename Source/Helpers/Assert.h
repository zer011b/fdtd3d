#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

/*
 * This function is used to exit and debugging purposes.
 */
void program_fail ();

/*
 * Indicates program point, which should not be reached.
 */
#if PRINT_MESSAGE
#define UNREACHABLE \
{ \
  printf ("Unreachable executed at %s:%d.\n", __FILE__, __LINE__); \
  program_fail (); \
}
#else /* PRINT_MESSAGE */
#define UNREACHABLE \
{ \
  program_fail (); \
}
#endif /* !PRINT_MESSAGE */

/*
 * Unconditional assert with message.
 */
#if PRINT_MESSAGE
#define ASSERT_MESSAGE(x) \
{ \
  printf ("Assert '%s' at %s:%d.\n", x, __FILE__, __LINE__); \
  program_fail (); \
}
#else /* PRINT_MESSAGE */
#define ASSERT_MESSAGE(x) \
{ \
  program_fail (); \
}
#endif /* !PRINT_MESSAGE */

/*
 * Conditional assert with default message.
 */
#if PRINT_MESSAGE
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    printf ("Assert at %s:%d.\n", __FILE__, __LINE__); \
    program_fail (); \
  } \
}
#else /* PRINT_MESSAGE */
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    program_fail (); \
  } \
}
#endif /* !PRINT_MESSAGE */

/*
 * Debug printf
 */
#if PRINT_MESSAGE
#define DPRINTF(...) \
{ \
    printf (__VA_ARGS__); \
}
#else /* PRINT_MESSAGE */
#define DPRINTF(...)
#endif /* !PRINT_MESSAGE */

#endif /* ASSERT_H */

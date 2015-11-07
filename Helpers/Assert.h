#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

#define UNREACHABLE \
  { \
    printf ("Unreachable executed\n"); \
  }

#define ASSERT (x) \
  { \
    printf ("Assert: %s", x); \
  }

#endif /* ASSERT_H */

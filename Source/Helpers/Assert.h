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
#define DPRINTF(logLevel, ...) \
  { \
    if (solverSettings.getLogLevel () >= logLevel) \
    { \
      printf (__VA_ARGS__); \
    } \
  }
#else /* PRINT_MESSAGE */
#define DPRINTF(...)
#endif /* !PRINT_MESSAGE */

/*
 * Enum class for c++11 and not c++11 builds
 */
#ifdef CXX11_ENABLED
#define ENUM_CLASS(name, type, ...) \
  enum class name : type \
  { \
    __VA_ARGS__ \
  };
#else /* CXX11_ENABLED */
#define ENUM_CLASS(name, type, ...) \
  class name \
  { \
    public: \
    \
    enum Temp { __VA_ARGS__ }; \
    \
    name (Temp new_val) : temp (new_val) {} \
    \
    operator type () { return temp; } \
    \
  private: \
    Temp temp; \
  };
#endif /* !CXX11_ENABLED */

/*
 * String to number
 */
#ifdef CXX11_ENABLED
#define STOI(str) std::stoi (str)
#define STOF(str) std::stof (str)
#else /* CXX11_ENABLED */
#define STOI(str) atoi (str)
#define STOF(str) atof (str)
#endif /* !CXX11_ENABLED */

#define EXIT_OK 0x0
#define EXIT_BREAK_ARG_PARSING 0x1
#define EXIT_ERROR 0xa
#define EXIT_UNKNOWN_OPTION 0xb

#endif /* ASSERT_H */

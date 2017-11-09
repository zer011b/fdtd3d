#ifndef ASSERT_H
#define ASSERT_H

#include <cstdio>

/*
 * This function is used to exit and debugging purposes.
 */
extern void program_fail ();

/*
 * Printf used for logging
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

#ifdef ENABLE_ASSERTS
/*
 * Indicates program point, which should not be reached.
 */
#define UNREACHABLE \
{ \
  DPRINTF (LOG_LEVEL_NONE, "Unreachable executed at %s:%d.\n", __FILE__, __LINE__); \
  program_fail (); \
}

/*
 * Unconditional assert with message.
 */
#define ASSERT_MESSAGE(x) \
{ \
  DPRINTF (LOG_LEVEL_NONE, "Assert '%s' at %s:%d.\n", x, __FILE__, __LINE__); \
  program_fail (); \
}

/*
 * Conditional assert with default message.
 */
#define ASSERT(x) \
{ \
  if (!(x)) \
  { \
    DPRINTF (LOG_LEVEL_NONE, "Assert at %s:%d.\n", __FILE__, __LINE__); \
    program_fail (); \
  } \
}
#else /* ENABLE_ASSERTS */
#define UNREACHABLE
#define ASSERT_MESSAGE(x)
#define ASSERT(x)
#endif /* !ENABLE_ASSERTS */

#define ALWAYS_ASSERT(x) \
{ \
  if (!(x)) \
  { \
    DPRINTF (LOG_LEVEL_NONE, "Assert at %s:%d.\n", __FILE__, __LINE__); \
    program_fail (); \
  } \
}

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

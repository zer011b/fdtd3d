#ifndef CLOCK_H
#define CLOCK_H

#include "Assert.h"
#include "FieldValue.h"
#include "Settings.h"

#ifdef MPI_CLOCK
#ifndef PARALLEL_GRID
#error Parallel Grid should be used with mpi clock
#endif /* !PARALLEL_GRID */
#include <mpi.h>
typedef FPValue ClockValue;
#else /* MPI_CLOCK */
typedef timespec ClockValue;
#endif /* !MPI_CLOCK */

/**
 * Clock corresponding to some moment of time in past. Diff of two such clocks could be used to measure elapsed time
 */
class Clock
{
  /**
   * Value of clock
   */
  ClockValue value;

public:

  /**
   * Get value of clock
   *
   * @return value of clock
   */
  const ClockValue & getVal () const
  {
    ASSERT (!isZero ());

    return value;
  } /* getVal */

  /**
   * Set new value of clock
   */
  void setVal (const ClockValue &val) /**< new value of clock */
  {
    value = val;
  } /* setVal */

  /**
   * Get floating point value of clock in seconds
   *
   * @return floating point value of clock in seconds
   */
  FPValue getFP () const
  {
    ASSERT (!isZero ());

#ifdef MPI_CLOCK
    FPValue val = value;
#else /* MPI_CLOCK */
    FPValue val = FPValue (value.tv_sec) + FPValue (value.tv_nsec) / FPValue (1000000000);
#endif /* !MPI_CLOCK */

    return val;
  } /* getFP */

  /**
   * Calculate sum of two clocks
   *
   * @return new clock with value of sum of two clock
   */
  Clock operator+ (const Clock &rhs) const /**< operand */
  {
    Clock clock;

#ifdef MPI_CLOCK
    clock.value = value + rhs.value;
#else /* MPI_CLOCK */
    timespec_sum (&value, &rhs.value, &clock.value);
#endif /* !MPI_CLOCK */

    return clock;
  } /* operator+ */

  /**
   * Calculate sum of two clocks
   *
   * @return new clock with value of sum of two clock
   */
  Clock & operator+= (const Clock &rhs) /**< operand */
  {
#ifdef MPI_CLOCK
    value += rhs.value;
#else /* MPI_CLOCK */
    timespec_sum (&value, &rhs.value, &value);
#endif /* !MPI_CLOCK */

    return *this;
  } /* operator+= */

  /**
   * Calculate diff of two clocks
   *
   * @return new clock with value of diff of two clock
   */
  Clock operator- (const Clock &rhs) const /**< operand */
  {
    Clock clock;

    ASSERT (operator>=(rhs));

#ifdef MPI_CLOCK
    clock.value = value - rhs.value;
#else /* MPI_CLOCK */
    timespec_diff (&value, &rhs.value, &clock.value);
#endif /* !MPI_CLOCK */

    return clock;
  } /* operator- */

  /**
   * Calculate >= condition
   *
   * @return true, if this clock is greater or equal than operand
   */
  bool operator>= (const Clock &rhs) const /**< operand */
  {
#ifdef MPI_CLOCK
    return value >= rhs.value;
#else /* MPI_CLOCK */
    int res = timespec_cmp (&value, &rhs.value);
    return res >= 0;
#endif /* !MPI_CLOCK */
  } /* operator>= */

  /**
   * Calculate > condition
   *
   * @return true, if this clock is greater than operand
   */
  bool operator> (const Clock &rhs) const /**< operand */
  {
#ifdef MPI_CLOCK
    return value > rhs.value;
#else /* MPI_CLOCK */
    int res = timespec_cmp (&value, &rhs.value);
    return res > 0;
#endif /* !MPI_CLOCK */
  } /* operator> */

  /**
   * Calculate <= condition
   *
   * @return true, if this clock is less or equal than operand
   */
  bool operator<= (const Clock &rhs) const /**< operand */
  {
#ifdef MPI_CLOCK
    return value <= rhs.value;
#else /* MPI_CLOCK */
    int res = timespec_cmp (&value, &rhs.value);
    return res <= 0;
#endif /* !MPI_CLOCK */
  } /* operator<= */

  /**
   * Calculate < condition
   *
   * @return true, if this clock is less than operand
   */
  bool operator< (const Clock &rhs) const /**< operand */
  {
#ifdef MPI_CLOCK
    return value < rhs.value;
#else /* MPI_CLOCK */
    int res = timespec_cmp (&value, &rhs.value);
    return res < 0;
#endif /* !MPI_CLOCK */
  } /* operator< */

  /**
   * Check whether clock is zero
   *
   * @return true if clock is zero
   */
  bool isZero () const
  {
#ifdef MPI_CLOCK
    return value == FPValue (0);
#else /* MPI_CLOCK */
    return value.tv_sec == 0 && value.tv_nsec == 0;
#endif /* !MPI_CLOCK */
  } /* isZero */

  /**
   * Print clock to console
   */
  void print ()
  {
#ifdef MPI_CLOCK
    printf ("Clock: %f seconds.\n", value);
#else /* MPI_CLOCK */
    printf ("Clock: %lu seconds, %lu nanoseconds.\n", (uint64_t) value.tv_sec, (uint64_t) value.tv_nsec);
#endif /* !MPI_CLOCK */
  } /* print */

  /**
   * Constructor, which doesn't initialize value with moment of time. Use Clock::getClock to create new clock.
   */
  Clock ()
  {
#ifdef MPI_CLOCK
    value = 0;
#else /* MPI_CLOCK */
    value.tv_sec = 0;
    value.tv_nsec = 0;
#endif /* !MPI_CLOCK */
  } /* Clock */

  /**
   * Destructor
   */
  ~Clock () {}

public:

  /**
   * Static constructor of clock corresponding to some moment of time
   */
  static Clock getNewClock ()
  {
    Clock clock;

#ifdef MPI_CLOCK
    clock.value = MPI_Wtime ();
#else /* MPI_CLOCK */
    int status = clock_gettime (CLOCK_MONOTONIC, &clock.value);
    ASSERT (status == 0);
#endif /* !MPI_CLOCK */

    return clock;
  } /* getNewClock */

  /**
   * Calculate average of two clocks
   */
  static Clock average (const Clock &lhs, const Clock &rhs)
  {
    Clock clock;

#ifdef MPI_CLOCK
    clock.value = (lhs.value + rhs.value) / FPValue (2);
#else /* MPI_CLOCK */
    timespec_avg (&lhs.value, &rhs.value, &clock.value);
#endif /* !MPI_CLOCK */

    return clock;
  } /* average */

private:

#ifndef MPI_CLOCK
  /**
   * Calculate difference of two moments in time
   */
  static void timespec_diff (const struct timespec *second, /**< second moment */
                             const struct timespec *first, /**< first moment */
                             struct timespec *result) /**< out: difference of two moments in time */
  {
    if ((second->tv_nsec - first->tv_nsec) < 0)
    {
      result->tv_sec = second->tv_sec - first->tv_sec - 1;
      result->tv_nsec = second->tv_nsec - first->tv_nsec + 1000000000;
    }
    else
    {
      result->tv_sec = second->tv_sec - first->tv_sec;
      result->tv_nsec = second->tv_nsec - first->tv_nsec;
    }
  } /* timespec_diff */

  /**
   * Calculate sum of two time diffs
   */
  static void timespec_sum (const struct timespec *second, /**< second moment */
                            const struct timespec *first, /**< first moment */
                            struct timespec *result) /**< out: difference of two moments in time */
  {
    result->tv_sec = first->tv_sec + second->tv_sec;
    result->tv_nsec = first->tv_nsec + second->tv_nsec;

    if (result->tv_nsec >= 1000000000)
    {
      result->tv_nsec -= 1000000000;
      result->tv_sec += 1;
    }
  } /* timespec_sum */

  /**
   * Calculate average value of two time diffs
   */
  static void timespec_avg (const struct timespec *second, /**< second moment */
                            const struct timespec *first, /**< first moment */
                            struct timespec *result) /**< out: average of two moments in time */
  {
    result->tv_sec = (first->tv_sec + second->tv_sec) / 2;
    result->tv_nsec = (first->tv_nsec + second->tv_nsec) / 2;
  } /* timespec_avg */

  /**
   * Compare two moment in time
   *
   * @return -1, if second is greater
   *          0, if both are equal
   *          1, if first is greater
   */
  static int timespec_cmp (const struct timespec *first, /**< first moment */
                           const struct timespec *second) /**< second moment */
  {
    if (first->tv_sec > second->tv_sec
        || (first->tv_sec == second->tv_sec && first->tv_nsec > second->tv_nsec))
    {
      return 1;
    }
    else if (first->tv_sec == second->tv_sec && first->tv_nsec == second->tv_nsec)
    {
      return 0;
    }
    else
    {
      return -1;
    }
  } /* timespec_diff */
#endif /* !MPI_CLOCK */

}; /* Clock */

#endif /* !CLOCK_H */

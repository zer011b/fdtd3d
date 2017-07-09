#ifndef PARALLEL_GRID_H
#define PARALLEL_GRID_H

#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

#include <mpi.h>

/**
 * Type of buffer of values
 */
typedef std::vector<FieldValue> VectorBufferValues;

/**
 * Type of vector of buffers
 */
typedef std::vector<VectorBufferValues> VectorBuffers;

/**
 * Type of vector of requests
 */
typedef std::vector<MPI_Request> VectorRequests;

/**
 * Parallel grid class
 *
 * Dimension of grid is set during build. Virtual topology of grid is also specified during build.
 *
 * Possible combinations:
 *   - 1D grid (has only one dimension and Ox axis):
 *     - Ox axis is spread through all computational nodes. If grid size is N, and number of computational nodes is P,
 *       then grid size per computational node is N/P. In case P is not divider of N, the last node is assigned grid
 *       with size (N - (N/P)*(P - 1)). See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                   ___ ___ ___       ___
 *       Ox axis -> |___|___|___| ... |___|
 *
 *   - 2D grid (has two dimensions and Ox,Oy axis):
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread.
 *       If grid size is N*M, and number of computational nodes is P, then grid size per computational node is (N/P)*M.
 *       In case P is not divider of N, the right border node by Ox axis is assigned grid with size with remainder
 *       (N - (N/P)*(P - 1))*M. See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                   ___ ___ ___       ___
 *                  |   |   |   | ... |   |
 *                  |   |   |   | ... |   |
 *       Ox axis -> |___|___|___| ... |___|

 *     - Ox axis is not spread, Oy axis is spread through all computational nodes.
 *       If grid size is N*M, and number of computational nodes is P, then grid size per computational node is N*(M/P).
 *       In case P is not divider of M, the right border node by Oy axis is assigned grid with size with remainder
 *       N*(M - (M/P)*(P - 1)). See PARALLEL_BUFFER_DIMENSION_1D_Y.
 *                   _____________________
 *                  |_____________________|
 *                  |_____________________|
 *                   .....................
 *                   _____________________
 *       Ox axis -> |_____________________|
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes.
 *       If grid size is N*M, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*(M/Q). In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*(M/Q), in case Q is not divider of M the right border node by Oy axis is assigned
 *       grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1)), in case both P is not divider of N and
 *       Q is not divider of M, the right corner border node by Ox and Oy axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_XY.
 *                   ___ ___ ___       ___
 *                  |___|___|___| ... |___|
 *                  |___|___|___| ... |___|
 *                   .....................
 *                   ___ ___ ___  ...  ___
 *       Ox axis -> |___|___|___| ... |___|
 *
 *   - 3D grid (has three dimensions and Ox,Oy,Oz axis):
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread, Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       (N/P)*M*K. In case P is not divider of N, the right border node by Ox axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*M*K. See PARALLEL_BUFFER_DIMENSION_1D_X.
 *                     ___ ___ ____          ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ / |  ...  /__ / |
 *                  |   |   |   |  |  ... |   |  |
 *                  |   |   |   | /   ... |   | /
 *       Ox axis -> |___|___|___|/    ... |___|/
 *
 *     - Ox axis is not spread, Oy axis is spread through all computational nodes, Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       N*(M/P)*K. In case P is not divider of M, the right border node by Oy axis is assigned grid with size
 *       with remainder N*(M - (M/P)*(P - 1))*K. See PARALLEL_BUFFER_DIMENSION_1D_Y.
 *                     __________________________
 *                    /                         /|
 *                   /________________________ //
 *                  |_________________________|/
 *                   .........................
 *                     __________________________
 *                    /                         /|
 *                   /________________________ //|
 *                  |_________________________|//
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is not spread, Oy axis is not spread, Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P, then grid size per computational node is
 *       N*(M/P)*K. In case P is not divider of M, the right border node by Oy axis is assigned grid with size
 *       with remainder N*(M - (M/P)*(P - 1))*K. See PARALLEL_BUFFER_DIMENSION_1D_Z.
 *
 *                            __________________________
 *                           /                         /|
 *                          /________________________ / |
 *                         |                         |  |
 *                         | ....................... |  |
 *                       __|_______________________  | /
 *                      /                         /|_|/
 *                     /________________________ / |
 *                    /                         /| |
 *                   /________________________ / | |
 *                  |                         |  |/
 *                  |                         |  /
 *                  |                         | /
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes,
 *       Oz axis is not spread.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*(M/Q)*K. In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*(M/Q)*K, in case Q is not divider of M the right border node by Oy axis is
 *       assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*K, in case both P is not divider of N and
 *       Q is not divider of M the right corner border node by Ox and Oy axis is assigned grid with
 *       size with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*K. See PARALLEL_BUFFER_DIMENSION_2D_XY.
 *                     ___ ___ ____          ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ //|  ...  /__ //|
 *                  |___|___|___|//   ... |___|//
 *                  |___|___|___|/    ... |___|/
 *                   ............................
 *                     ____ ___ ___   ...    ____
 *                    /   /   /   /|  ...   /   /|
 *                   /__ /__ /__ //   ...  /__ //
 *       Ox axis -> |___|___|___|/    ... |___|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is not spread,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       (N/P)*M*(K/Q). In case P is not divider of N, the right border node by Ox axis is assigned grid with size with
 *       remainder (N - (N/P)*(P - 1))*M*(K/Q), in case Q is not divider of K the right border node by Oz axis is
 *       assigned grid with size with remainder (N/P)*M*(K - (K/Q)*(Q - 1)), in case both P is not divider of N and
 *       Q is not divider of K the right corner border node by Ox and Oz axis is assigned grid with
 *       size with remainder (N - (N/P)*(P - 1))*M*(K - (K/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_XZ.
 *
 *                         ___ ____ ___          ____
 *                        /__ /__ /__ /|    .   /__ /|
 *                       |   |   |   | |   .   |   | |
 *                     __|_ _|_ _|_  | |  .  __|_  | |
 *                    /___/___/___/|_|/  .  /___/|_|/
 *                   /__ /__ /__ /||......./__ /||.
 *                  |   |   |   | ||   .  |   | ||
 *                  |   |   |   | /   .   |   | /
 *       Ox axis -> |___|___|___|/   .    |___|/
 *
 *     - Ox axis is not spread, Oy axis is spread through all computational nodes,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q, then grid size per computational node is
 *       N*(M/P)*(K/Q). In case P is not divider of M, the right border node by Oy axis is assigned grid with size with
 *       remainder N*(M - (M/P)*(P - 1))*(K/Q), in case Q is not divider of K the right border node by Oz axis is
 *       assigned grid with size with remainder N*(M/P)*(K - (K/Q)*(Q - 1)), in case both P is not divider of M and
 *       Q is not divider of K the right corner border node by Oy and Oz axis is assigned grid with
 *       size with remainder N*(M - (M/P)*(P - 1))*(K - (K/Q)*(Q - 1)). See PARALLEL_BUFFER_DIMENSION_2D_YZ.
 *                               __________________________
 *                              /________________________ /|
 *                             |_________________________|/
 *                                                      .
 *                    __________________________ _______.__
 *                   /________________________ /|_______. /|
 *                  |_________________________|/________.|/|
 *                    .        |________________________.|/
 *                    .        .                       .
 *                    .______.__________________     .
 *                   /._______________________ /|  .
 *                  |_________________________|/|.
 *       Ox axis -> |_________________________|/
 *
 *     - Ox axis is spread through all computational nodes, Oy axis is spread through all computational nodes,
 *       Oz axis is spread through all computational nodes.
 *       If grid size is N*M*K, and number of computational nodes is P*Q*R, then grid size per computational node is
 *       (N/P)*(M/Q)*(K/R). In case P is not divider of N, the right border node by Ox axis is assigned grid with size
 *       with remainder (N - (N/P)*(P - 1))*(M/Q)*(K/R), in case Q is not divider of M the right border node by Oy axis
 *       is assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*(K/R), in case R is not divider of K
 *       the right border node by Oz axis is assigned grid with size with remainder (N/P)*(M/Q)*(K - (K/R)*(R - 1)),
 *       in case both P is not divider of N and Q is not divider of M the right corner border node by Ox and Oy axis is
 *       assigned grid with size with remainder (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*(K/R),
 *       in case both P is not divider of N and R is not divider of K the right corner border node by Ox and Oz axis is
 *       assigned grid with size with remainder (N - (N/P)*(P - 1))*(M/Q)*(K - (K/R)*(R - 1)),
 *       in case both Q is not divider of M and R is not divider of K the right corner border node by Oy and Oz axis is
 *       assigned grid with size with remainder (N/P)*(M - (M/Q)*(Q - 1))*(K - (K/R)*(R - 1)),
 *       in case P is not divider of N, Q is not divider of M and R is not divider of K, the right corner border node
 *       by Ox, Oy and Oz axis is assigned grid with size with remainder
 *       (N - (N/P)*(P - 1))*(M - (M/Q)*(Q - 1))*(K - (K/R)*(R - 1)). See PARALLEL_BUFFER_DIMENSION_3D_XYZ.
 *
 * On all pictures above axis are:
 *       Oy    Oz
 *       +    +
 *       |   /
 *       |  /
 *       | /
 *       |/_____________+ Ox
 *
 * All grids corresponding to computational nodes have buffers only on the sides where nodes have neighbours.
 */
class ParallelGrid: public ParallelGridBase
{
private:

  /**
   * Static data shared between all parallel grid on this computational node
   */
  static ParallelGridCore *parallelGridCore;

private:

  /**
   * =======================================
   * Parameters corresponding to parallelism
   * =======================================
   */

  /**
   * Array of coordinate in grid from which to start send values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> sendStart;

  /**
   * Array of coordinate in grid until which to send values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> sendEnd;

  /**
   * Array of coordinate in grid from which to start saving received values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> recvStart;

  /**
   * Array of coordinate in grid until which to save received values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> recvEnd;

  /**
   * =======================================
   * Parameters corresponding to data in grid
   * =======================================
   */

  /**
   * Total size of grid (size, which is specified at its declaration)
   */
  ParallelGridCoordinate totalSize;

  /**
   * Size of grid for current node without buffers (raw data which is assigned to current computational node)
   */
  ParallelGridCoordinate currentSize;

  /**
   * Size of grid per node which is used for all buffers except at right border ones (see ParallelGrid)
   */
  ParallelGridCoordinate coreCurrentSize;

  /**
   * Absolute start position of chunk of current node
   */
  ParallelGridCoordinate posStart;

  /**
   * Size of buffer zone
   */
  ParallelGridCoordinate bufferSize;

  /**
   * Send buffers to send values from it
   *
   * TODO: remove copy to this buffer before send
   */
  VectorBuffers buffersSend;

  /**
   * Receive buffers to receive values into
   *
   * TODO: remove copy from this buffer after receive
   */
  VectorBuffers buffersReceive;

  /**
   * Step at which to perform share operations for synchronization of computational nodes
   */
  time_step shareStep;

private:

  void SendRawBuffer (BufferPosition, int);
  void ReceiveRawBuffer (BufferPosition, int);
  void SendReceiveRawBuffer (BufferPosition, int, BufferPosition, int);
  void SendReceiveBuffer (BufferPosition);
  void SendReceive ();

  void ParallelGridConstructor ();

  void InitBuffers ();

  void SendReceiveCoordinatesInit ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_X ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_Y ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_Z ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_XY ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_YZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_XZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit3D_XYZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  void initializeStartPosition ();

public:

  ParallelGrid (const ParallelGridCoordinate &,
                const ParallelGridCoordinate &,
                time_step,
                ParallelGridCoordinate,
                ParallelGridCoordinate,
                const char * = "unnamed");

  virtual void nextTimeStep () CXX11_OVERRIDE;

  void nextShareStep ();
  void zeroShareStep ();
  void share ();

  virtual ParallelGridCoordinate getComputationEnd (ParallelGridCoordinate) const CXX11_OVERRIDE;
  virtual ParallelGridCoordinate getComputationStart (ParallelGridCoordinate) const CXX11_OVERRIDE;

  ParallelGridCoordinate getStartPosition () const;
  ParallelGridCoordinate getChunkStartPosition () const;
  ParallelGridCoordinate getTotalPosition (ParallelGridCoordinate);
  ParallelGridCoordinate getRelativePosition (ParallelGridCoordinate);

  FieldPointValue *getFieldPointValueByAbsolutePos (const ParallelGridCoordinate &);
  FieldPointValue *getFieldPointValueOrNullByAbsolutePos (const ParallelGridCoordinate &);

  /**
   * Getter for total size of grid
   *
   * @return total size of grid
   */
  ParallelGridCoordinate getTotalSize () const
  {
    return totalSize;
  } /* getTotalSize */

  /**
   * Getter for size of grid assigned to current node
   *
   * @return size of grid assigned to current node
   */
  ParallelGridCoordinate getCurrentSize () const
  {
    return currentSize;
  } /* getCurrentSize */

  /**
   * Getter for size of buffer
   *
   * @return size of buffer
   */
  ParallelGridCoordinate getBufferSize () const
  {
    return bufferSize;
  } /* getBufferSize */

  void initBufferOffsets (grid_coord &, grid_coord &, grid_coord &, grid_coord &, grid_coord &, grid_coord &) const;

  Grid<ParallelGridCoordinate> gatherFullGrid () const;

public:

  static void initializeParallelCore (ParallelGridCore *);
  static ParallelGridCore *getParallelCore ();

}; /* ParallelGrid */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_H */

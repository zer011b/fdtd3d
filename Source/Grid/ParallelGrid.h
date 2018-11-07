#ifndef PARALLEL_GRID_H
#define PARALLEL_GRID_H

#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

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
 * All grids corresponding to computational nodes have buffers only on the sides where nodes have neighbors.
 */
class ParallelGrid: public ParallelGridBase
{
private:

  /**
   * Static data shared between all parallel grids on this computational node
   */
  static ParallelGridCore *parallelGridCore;

private:

#ifdef DEBUG_INFO
  /**
   * Coordinate types for ParallelGridCoordinate, corresponding to this ParallelGrid.
   *
   * Note: This couldn't be placed in ParallelGridCore, as same ParallelGridCore could be used for different virtual
   *       topologies of same dimension, for example, for 2D-XY and for 2D-XZ, however, coordinate type must be
   *       different for them.
   */
  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;
#endif /* DEBUG_INFO */

  /**
   * ========================================
   * Parameters corresponding to parallelism
   * ========================================
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
   * ========================================
   * Parameters corresponding to data in grid
   * ========================================
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

  void initializeStartPosition (ParallelGridCoordinate);
  void gatherStartPosition ();

public:

  ParallelGrid (const ParallelGridCoordinate &,
                const ParallelGridCoordinate &,
                time_step,
                ParallelGridCoordinate,
                const char * = "unnamed");

  virtual ~ParallelGrid () {}

  virtual void nextTimeStep () CXX11_OVERRIDE;

  /**
   * Increase share step
   */
  void nextShareStep ()
  {
    ++shareStep;
  } /* nextShareStep */

  time_step getShareStep ()
  {
    return shareStep;
  }

  /**
   * Set share step to zero
   */
  void zeroShareStep ()
  {
    shareStep = 0;
  } /* zeroShareStep */

  void share ();

  /**
   * Get first coordinate from which to perform computations at current step
   *
   * @return first coordinate from which to perform computations at current step
   */
  virtual ParallelGridCoordinate getComputationStart
    (ParallelGridCoordinate diffPosStart) const CXX11_OVERRIDE /**< layout coordinate modifier */
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffX = diffPosStart.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffY = diffPosStart.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord diffZ = diffPosStart.get3 ();
#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasL ())
    {
      diffX = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasD ())
    {
      diffY = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasB ())
    {
      diffZ = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord px = diffX;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord py = diffY;
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord pz = diffZ;
#endif /* GRID_3D */

#ifdef GRID_1D
    return ParallelGridCoordinate (px
#ifdef DEBUG_INFO
                                   , ct1
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_1D */
#ifdef GRID_2D
    return ParallelGridCoordinate (px, py
#ifdef DEBUG_INFO
                                   , ct1, ct2
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_2D */
#ifdef GRID_3D
    return ParallelGridCoordinate (px, py, pz
#ifdef DEBUG_INFO
                                   , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_3D */
  } /* getComputationStart */

  /**
   * Get last coordinate until which to perform computations at current step
   *
   * @return last coordinate until which to perform computations at current step
   */
  virtual ParallelGridCoordinate getComputationEnd
    (ParallelGridCoordinate diffPosEnd) const CXX11_OVERRIDE /**< layout coordinate modifier */
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffX = diffPosEnd.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffY = diffPosEnd.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord diffZ = diffPosEnd.get3 ();
#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasR ())
    {
      diffX = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasU ())
    {
      diffY = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasF ())
    {
      diffZ = shareStep + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord px = ParallelGridBase::getSize ().get1 () - diffX;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord py = ParallelGridBase::getSize ().get2 () - diffY;
#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D
    grid_coord pz = ParallelGridBase::getSize ().get3 () - diffZ;
#endif /* GRID_3D */

#ifdef GRID_1D
    return ParallelGridCoordinate (px
#ifdef DEBUG_INFO
                                   , ct1
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_1D */
#ifdef GRID_2D
    return ParallelGridCoordinate (px, py
#ifdef DEBUG_INFO
                                   , ct1, ct2
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_2D */
#ifdef GRID_3D
    return ParallelGridCoordinate (px, py, pz
#ifdef DEBUG_INFO
                                   , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_3D */
  } /* ParallelGrid::getComputationEnd */

  /**
   * Get absolute position corresponding to first value in grid for current computational node (considering buffers)
   *
   * @return absolute position corresponding to first value in grid for current computational node (considering buffers)
   */
  ParallelGridCoordinate getStartPosition () const
  {
    return posStart;
  } /* getStartPosition */

  /**
   * Get absolute position corresponding to first value in grid for current computational node (not considering buffers)
   *
   * @return absolute position corresponding to first value in grid for current computational node (not considering buffers)
   */
  ParallelGridCoordinate getChunkStartPosition () const
  {
    grid_coord left_coord, right_coord;
    grid_coord down_coord, up_coord;
    grid_coord back_coord, front_coord;

    initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

#ifdef GRID_1D
    return posStart + GridCoordinate1D (left_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_1D */

#ifdef GRID_2D
    return posStart + GridCoordinate2D (left_coord, down_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_2D */

#ifdef GRID_3D
    return posStart + GridCoordinate3D (left_coord, down_coord, back_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
                                        , getSize ().getType3 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_3D */
  } /* getChunkStartPosition */

  /**
   * Get total position in grid from relative position for current computational node
   *
   * @return total position in grid from relative position for current computational node
   */
  virtual ParallelGridCoordinate getTotalPosition (ParallelGridCoordinate pos) const CXX11_OVERRIDE /**< relative
                                                                                                     *   position for
                                                                                                     *   current
                                                                                                     *   computational
                                                                                                     *   node */
  {
    ParallelGridCoordinate posStart = getStartPosition ();

    return posStart + pos;
  } /* getTotalPosition */

  /**
   * Get relative position for current computational node from total position
   *
   * @return relative position for current computational node from total position
   */
  virtual ParallelGridCoordinate getRelativePosition (ParallelGridCoordinate pos) const CXX11_OVERRIDE /**< total
                                                                                                        *   position in
                                                                                                        *   grid */
  {
    ParallelGridCoordinate posStart = getStartPosition ();

    ASSERT (pos >= posStart);

    return pos - posStart;
  } /* ParallelGrid::getRelativePosition */

  /**
   * Get field point value at absolute coordinate in grid
   *
   * @return field point value
   */
  virtual FieldPointValue * getFieldPointValueByAbsolutePos
    (const ParallelGridCoordinate &absPosition) CXX11_OVERRIDE /**< absolute coordinate in grid */
  {
    return getFieldPointValue (getRelativePosition (absPosition));
  } /* getFieldPointValueByAbsolutePos */

  /**
   * Get field point value at absolute coordinate in grid. If current node does not contain this coordinate, return NULLPTR
   *
   * @return field point value or NULLPTR
   */
  virtual FieldPointValue * getFieldPointValueOrNullByAbsolutePos
    (const ParallelGridCoordinate &absPosition) CXX11_OVERRIDE /**< absolute coordinate in grid */
  {
    if (!hasValueForCoordinate (absPosition))
    {
      return NULLPTR;
    }

    ParallelGridCoordinate relPosition = getRelativePosition (absPosition);

    return getFieldPointValue (relPosition);
  } /* getFieldPointValueOrNullByAbsolutePos */

  /**
   * Check whether current node has value for coordinate
   *
   * @return true, if current node contains value for coordinate, false otherwise
   */
  bool hasValueForCoordinate (const ParallelGridCoordinate &position) /**< coordinate of value to check */
  {
    ParallelGridCoordinate posStart = getStartPosition ();
    ParallelGridCoordinate posEnd = posStart + getSize ();

    if (!(position >= posStart)
        || !(position < posEnd))
    {
      return false;
    }

    return true;
  } /* hasValueForCoordinate */

  /**
   * Getter for total size of grid
   *
   * @return total size of grid
   */
  virtual ParallelGridCoordinate getTotalSize () const CXX11_OVERRIDE
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

  /**
   * Initialize buffer offsets for computational node
   */
  void initBufferOffsets (grid_coord &left_coord, /**< out: left buffer size */
                          grid_coord &right_coord, /**< out: right buffer size */
                          grid_coord &down_coord, /**< out: down buffer size */
                          grid_coord &up_coord, /**< out: up buffer size */
                          grid_coord &back_coord, /**< out: back buffer size */
                          grid_coord &front_coord) const /**< out: front buffer size */
  {
    left_coord = 0;
    right_coord = 0;
    down_coord = 0;
    up_coord = 0;
    back_coord = 0;
    front_coord = 0;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    left_coord = 0;
    if (parallelGridCore->getNodeForDirection (LEFT) != PID_NONE)
    {
      left_coord = bufferSize.get1 ();
    }
    else
    {
      ASSERT (left_coord == 0);
    }

    right_coord = 0;
    if (parallelGridCore->getNodeForDirection (RIGHT) != PID_NONE)
    {
      right_coord = bufferSize.get1 ();
    }
    else
    {
      ASSERT (right_coord == 0);
    }
#else /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
         PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
    left_coord = 0;
    right_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_X && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_XZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    down_coord = 0;
    if (parallelGridCore->getNodeForDirection (DOWN) != PID_NONE)
    {
      down_coord = bufferSize.get2 ();
    }
    else
    {
      ASSERT (down_coord == 0);
    }

    up_coord = 0;
    if (parallelGridCore->getNodeForDirection (UP) != PID_NONE)
    {
      up_coord = bufferSize.get2 ();
    }
    else
    {
      ASSERT (up_coord == 0);
    }
#else /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
         PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
    down_coord = 0;
    up_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_Y && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_YZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    back_coord = 0;
    if (parallelGridCore->getNodeForDirection (BACK) != PID_NONE)
    {
      back_coord = bufferSize.get3 ();
    }
    else
    {
      ASSERT (back_coord == 0);
    }

    front_coord = 0;
    if (parallelGridCore->getNodeForDirection (FRONT) != PID_NONE)
    {
      front_coord = bufferSize.get3 ();
    }
    else
    {
      ASSERT (front_coord == 0);
    }
#else /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
         PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
    back_coord = 0;
    front_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_Y && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_YZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  } /* initBufferOffsets */

  ParallelGridBase *gatherFullGrid () const;
  ParallelGridBase *gatherFullGridPlacement (ParallelGridBase *) const;

#ifdef DYNAMIC_GRID
  void Resize (ParallelGridCoordinate);
#endif /* DYNAMIC_GRID */

  /**
   * Check whether position corresponds to left buffer or not
   *
   * @return true, if position is in left buffer, false, otherwise
   */
  bool isBufferLeftPosition (ParallelGridCoordinate pos) /**< position to check */
  {
    ASSERT (pos < totalSize);
    ParallelGridCoordinate chunkStart = getChunkStartPosition ();

    if (pos >= chunkStart)
    {
      return false;
    }
    else
    {
      return true;
    }
  } /* isBufferLeftPosition */

  /**
   * Check whether position corresponds to right buffer or not
   *
   * @return true, if position is in right buffer, false, otherwise
   */
  bool isBufferRightPosition (ParallelGridCoordinate pos) /**< position to check */
  {
    ASSERT (pos < totalSize);
    ParallelGridCoordinate chunkEnd = getChunkStartPosition () + currentSize;

    if (pos < chunkEnd)
    {
      return false;
    }
    else
    {
      return true;
    }
  } /* isBufferRightPosition */

  BufferPosition getBufferForPosition (ParallelGridCoordinate) const;

  /**
   * Getter for array of coordinate in grid from which to start send values corresponding to direction
   *
   * @return array of coordinate in grid from which to start send values corresponding to direction
   */
  const std::vector<ParallelGridCoordinate> &getSendStart () const
  {
    return sendStart;
  } /* getSendStart */

  /**
   * Getter for array of coordinate in grid until which to send values corresponding to direction
   *
   * @return array of coordinate in grid until which to send values corresponding to direction
   */
  const std::vector<ParallelGridCoordinate> &getSendEnd () const
  {
    return sendEnd;
  } /* getSendEnd */

  /**
   * Getter for array of coordinate in grid from which to start saving received values corresponding to direction
   *
   * @return array of coordinate in grid from which to start saving received values corresponding to direction
   */
  const std::vector<ParallelGridCoordinate> &getRecvStart () const
  {
    return recvStart;
  } /* getRecvStart */

  /**
   * Getter for array of coordinate in grid until which to save received values corresponding to direction
   *
   * @return array of coordinate in grid until which to save received values corresponding to direction
   */
  const std::vector<ParallelGridCoordinate> &getRecvEnd () const
  {
    return recvEnd;
  } /* getRecvEnd */

public:

  /**
   * Initialize parallel grid core
   */
  static void initializeParallelCore (ParallelGridCore *core) /**< new parallel grid core */
  {
    ASSERT (parallelGridCore == NULLPTR);

    parallelGridCore = core;
  } /* initializeParallelCore */

  /**
   * Get parallel grid core
   */
  static ParallelGridCore * getParallelCore ()
  {
    ASSERT (parallelGridCore != NULLPTR);
    return parallelGridCore;
  } /* getParallelCore */

}; /* ParallelGrid */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_H */

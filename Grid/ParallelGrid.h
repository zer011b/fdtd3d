#ifndef PARALLEL_GRID_H
#define PARALLEL_GRID_H

#include "Grid.h"

#ifdef PARALLEL_GRID

#include <mpi.h>

#ifdef GRID_1D
#define ParallelGridBase Grid<GridCoordinate1D>
#define ParallelGridCoordinate GridCoordinate1D
#endif /* GRID_1D */

#ifdef GRID_2D
#define ParallelGridBase Grid<GridCoordinate2D>
#define ParallelGridCoordinate GridCoordinate2D
#endif /* GRID_2D */

#ifdef GRID_3D
#define ParallelGridBase Grid<GridCoordinate3D>
#define ParallelGridCoordinate GridCoordinate3D
#endif /* GRID_3D */

/*
 * Parallel grid buffer types.
 */
#ifdef PARALLEL_GRID
enum BufferPosition
{
#define FUNCTION(X) X,
#include "BufferPosition.inc.h"
  PARALLEL_BUFFERS
#undef FUNCTION
};
#endif /* PARALLEL_GRID */

// Type for buffer of values.
typedef std::vector<FieldValue> VectorBufferValues;

// Type for vector of buffer
typedef std::vector<VectorBufferValues> VectorBuffers;

// Type for vector of buffer
typedef std::vector<MPI_Request> VectorRequests;

class ParallelGrid: public ParallelGridBase
{
  // Current node (process) identificator.
  int processId;

  // Overall count of nodes (processes).
  int totalProcCount;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  int nodeGridSizeX;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
  int nodeGridSizeY;
#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D
  int nodeGridSizeZ;
#endif /* GRID_3D */

  std::vector<ParallelGridCoordinate> sendStart;
  std::vector<ParallelGridCoordinate> sendEnd;
  std::vector<ParallelGridCoordinate> recvStart;
  std::vector<ParallelGridCoordinate> recvEnd;

  std::vector<BufferPosition> oppositeDirections;

  std::vector<int> directions;

  std::vector< std::pair<bool, bool> > doShare;

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int nodeGridSizeXY;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  int nodeGridSizeYZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  int nodeGridSizeXZ;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  int nodeGridSizeXYZ;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  // Size of current node without buffers.
  ParallelGridCoordinate totalSize;

  // Size of current node without buffers.
  ParallelGridCoordinate currentSize;

  // Size of buffer zone.
  ParallelGridCoordinate bufferSizeLeft;
  ParallelGridCoordinate bufferSizeRight;

  // Send/Receive buffers for independent send and receive
  VectorBuffers buffersSend;
  VectorBuffers buffersReceive;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasL;
  bool hasR;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasD;
  bool hasU;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasB;
  bool hasF;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  time_step shareStep;

private:

  // Raw send
  void SendRawBuffer (BufferPosition buffer, int processTo);

  // Raw receive
  void ReceiveRawBuffer (BufferPosition buffer, int processFrom);

  // Raw send/receive
  void SendReceiveRawBuffer (BufferPosition bufferSend, int processTo,
                             BufferPosition bufferReceive, int processFrom);

  void SendReceiveBuffer (BufferPosition bufferDirection);

  void SendReceive ();

  void ParallelGridConstructor (grid_iter numTimeStepsInBuild);
  void NodeGridInit ();
  ParallelGridCoordinate GridInit ();
  void InitDirections ();
  void InitBufferFlags ();
  void InitBuffers (grid_iter numTimeStepsInBuild);

  void SendReceiveCoordinatesInit ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  void SendReceiveCoordinatesInit1D_X ();
  int getNodeGridX ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  void SendReceiveCoordinatesInit1D_Y ();
  int getNodeGridY ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  void SendReceiveCoordinatesInit1D_Z ();
  int getNodeGridZ ();
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


#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  void FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& left, FieldValue alpha);
  void NodeGridInitInner (FieldValue& overall1, FieldValue& overall2,
                          int& nodeGridSize1, int& nodeGridSize2, int& left);
  void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                 grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  void FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left,
                                  FieldValue alpha, FieldValue betta);
  void NodeGridInitInner (FieldValue& overall1, FieldValue& overall2, FieldValue& overall3,
                          int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left);
  void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                 grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2,
                                 grid_coord& c3, int nodeGridSize3, bool has3, grid_coord size3);
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  BufferPosition getOpposite (BufferPosition direction);
  void getShare (BufferPosition direction, std::pair<bool, bool>& pair);

public:

  ParallelGrid (const ParallelGridCoordinate& totSize,
                const ParallelGridCoordinate& bufSizeL, const ParallelGridCoordinate& bufSizeR,
                const int process, const int totalProc, uint32_t step);

  // Switch to next time step.
  virtual void nextTimeStep ();

  void share ();

  ParallelGridCoordinate getBufferSize () const;

  // Get field point at coordinate in grid.
  FieldPointValue* getFieldPointValueAbsoluteIndex (const ParallelGridCoordinate& position);

  ParallelGridCoordinate getSize () const
  {
#ifdef GRID_1D
    return ParallelGridCoordinate (ParallelGridBase::getSize ().getX () - shareStep);
#endif /* GRID_1D */

#ifdef GRID_2D
    return ParallelGridCoordinate (ParallelGridBase::getSize ().getX () - shareStep,
                                   ParallelGridBase::getSize ().getY () - shareStep);
#endif /* GRID_2D */

#ifdef GRID_3D
    return ParallelGridCoordinate (ParallelGridBase::getSize ().getX () - shareStep,
                                   ParallelGridBase::getSize ().getY () - shareStep,
                                   ParallelGridBase::getSize ().getZ () - shareStep);
#endif /* GRID_3D */
  }

  ParallelGridCoordinate getStart () const
  {
    return ParallelGridCoordinate (shareStep);
  }
};

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_H */

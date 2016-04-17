#ifndef PARALLEL_GRID_H
#define PARALLEL_GRID_H

#if defined (PARALLEL_GRID)

#if defined (GRID_1D)
#include "Grid1D.h"
#define ParallelGridBase Grid1D
#endif

#if defined (GRID_2D)
#include "Grid2D.h"
#define ParallelGridBase Grid2D
#endif

#if defined (GRID_3D)
#include "Grid3D.h"
#define ParallelGridBase Grid3D
#endif

/*
 * Parallel grid buffer types.
 */
#if defined (PARALLEL_GRID)
enum BufferPosition
{
#define FUNCTION(X) X,
#include "BufferPosition.inc.h"
#undef FUNCTION
};
#endif

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
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  int nodeGridSizeY;
#endif
#if defined (GRID_3D)
  int nodeGridSizeZ;
#endif

  std::vector<GridCoordinate> sendStart;
  std::vector<GridCoordinate> sendEnd;
  std::vector<GridCoordinate> recvStart;
  std::vector<GridCoordinate> recvEnd;

  std::vector<BufferPosition> oppositeDirections;

  std::vector<int> directions;

  std::vector< std::pair<bool, bool> > doShare;

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int nodeGridSizeXY;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  int nodeGridSizeYZ;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  int nodeGridSizeXZ;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int nodeGridSizeXYZ;
#endif

  // Size of current node without buffers.
  GridCoordinate totalSize;

  // Size of current node without buffers.
  GridCoordinate currentSize;

  // Size of buffer zone.
  GridCoordinate bufferSizeLeft;
  GridCoordinate bufferSizeRight;

  // Send/Receive buffers for independent send and receive
  VectorBuffers buffersSend;
  VectorBuffers buffersReceive;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasL = false;
  bool hasR = false;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasD = false;
  bool hasU = false;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasB = false;
  bool hasF = false;
#endif

#endif /* PARALLEL_GRID */

private:

  #if defined (PARALLEL_GRID)
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
    GridCoordinate GridInit ();
    void InitDirections ();
    void InitBufferFlags ();
    void InitBuffers (grid_iter numTimeStepsInBuild);

    void SendReceiveCoordinatesInit ();

  #if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit1D_X ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit1D_Y ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit1D_Z ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit2D_XY ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit2D_YZ ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit2D_XZ ();
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void SendReceiveCoordinatesInit3D_XYZ ();
  #endif

  #if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
    void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1);
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
    void FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& left, FieldValue alpha);
    void NodeGridInitInner (FieldValue& overall1, FieldValue& overall2,
                            int& nodeGridSize1, int& nodeGridSize2, int& left);
    void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                   grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2);
  #endif
  #if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    void FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left,
                                    FieldValue alpha, FieldValue betta);
    void NodeGridInitInner (FieldValue& overall1, FieldValue& overall2, FieldValue& overall3,
                            int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left);
    void CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                   grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2,
                                   grid_coord& c3, int nodeGridSize3, bool has3, grid_coord size3);
  #endif

    BufferPosition getOpposite (BufferPosition direction);
    void getShare (BufferPosition direction, std::pair<bool, bool>& pair);

  #endif /* PARALLEL_GRID */

public:

  Grid (const GridCoordinate& totSize,
        const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
        const int process, const int totalProc, uint32_t step);

  #if defined (PARALLEL_GRID)
    void Share ();
  #endif /* PARALLEL_GRID */



};

#endif /* PARALLEL_GRID_H */

#include "Grid.h"

#include <cmath>

extern const char* BufferPositionNames[];

#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
Grid::NodeGridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());
  c2 = totalSize.getY ();
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
  nodeGridSizeZ = 1;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = totalProcCount;

  c1 = totalSize.getX ();
  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c3, nodeGridSizeZ, totalSize.getZ ());
#endif

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ);
#endif
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
Grid::NodeGridInit ()
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getY ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  FieldValue overall1 = (FieldValue) totalSize.getY ();
  FieldValue overall2 = (FieldValue) totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getZ ();
#endif

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  NodeGridInitInner (overall1, overall2, nodeGridSizeTmp1, nodeGridSizeTmp2, left);

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY (), c3, nodeGridSizeZ, totalSize.getZ ());
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c3, nodeGridSizeZ, totalSize.getZ ());
#endif

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
#endif
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::NodeGridInit ()
{
  if (totalProcCount < 8)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  grid_coord overall1 = totalSize.getX ();
  grid_coord overall2 = totalSize.getY ();
  grid_coord overall3 = totalSize.getZ ();

  FieldValue alpha = 0;
  FieldValue betta = 0;
  if (overall1 < overall2 && overall1 < overall3)
  {
    alpha = overall2 / overall1;
    betta = overall2 / overall1;
  }
  else if (overall2 < overall1 && overall2 < overall3)
  {
    alpha = overall1 / overall2;
    betta = overall3 / overall2;
  }
  else if (overall3 < overall1 && overall3 < overall2)
  {
    alpha = overall1 / overall3;
    betta = overall2 / overall3;
  }

  FieldValue cbrtVal = ((FieldValue) (totalProcCount)) / (alpha * betta);
  cbrtVal = cbrt (cbrtVal);

  if (cbrtVal <= 1)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  cbrtVal = round (cbrtVal);
  ASSERT (cbrtVal == floor (cbrtVal));

  grid_coord nodeGridSizeTmp1;
  grid_coord nodeGridSizeTmp2;

  if (overall1 > overall2)
  {
    nodeGridSizeTmp2 = (int) sqrtVal;
    nodeGridSizeTmp1 = totalProcCount / nodeGridSizeTmp2;
  }
  else
  {
    nodeGridSizeTmp1 = (int) sqrtVal;
    nodeGridSizeTmp2 = totalProcCount / nodeGridSizeTmp1;
  }

  int left = totalProcCount - nodeGridSizeTmp1 * nodeGridSizeTmp2;

  // Considerable. Could give up if only one left
  if (left > 0) /* left > 1 */
  {
    // Bad case, too many nodes left unused. Let's change proportion.
    bool find = true;
    bool direction1 = nodeGridSizeTmp1 > nodeGridSizeTmp2 ? true : false;
    while (find)
    {
      find = false;
      if (direction1 && nodeGridSizeTmp1 > 2)
      {
        find = true;
        --nodeGridSizeTmp1;
        nodeGridSizeTmp2 = totalProcCount / nodeGridSizeTmp1;
      }
      else if (!direction1 && nodeGridSizeTmp2 > 2)
      {
        find = true;
        --nodeGridSizeTmp2;
        nodeGridSizeTmp1 = totalProcCount / nodeGridSizeTmp2;
      }

      left = totalProcCount - nodeGridSizeTmp1 * nodeGridSizeTmp2;

      if (find && left == 0)
      {
        find = false;
      }
    }
  }

  ASSERT (nodeGridSizeTmp1 > 1 && nodeGridSizeTmp2 > 1);

  grid_coord c1;
  grid_coord c2;
  if ((processId + 1) % nodeGridSizeTmp1 != 0)
    c1 = totalSize.getX () / nodeGridSizeTmp1;
  else
    c1 = totalSize.getX () - (nodeGridSizeTmp1 - 1) * (totalSize.getX () / nodeGridSizeTmp1);

  if (processId < nodeGridSizeTmp1 * nodeGridSizeTmp2 - nodeGridSizeTmp1)
    c2 = totalSize.getY () / nodeGridSizeTmp2;
  else
    c2 = totalSize.getY () - (nodeGridSizeTmp2 - 1) * (totalSize.getY () / nodeGridSizeTmp2);

  currentSize = GridCoordinate (c1, c2);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = 1;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;
#endif

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
#endif
}
#endif




#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X ||
          PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#endif /* GRID_3D */

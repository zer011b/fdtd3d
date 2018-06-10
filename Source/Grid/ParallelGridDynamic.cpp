#ifdef DYNAMIC_GRID

/**
 * Resize parallel grid for current process at a new size
 */
void
ParallelGrid::Resize (ParallelGridCoordinate newCurrentNodeSize) /**< new size of chunk assigned to current process */
{
  /*
   * TODO: do not allocate each time
   */
  ParallelGridBase *totalGrid = gatherFullGrid ();

  ParallelGridCoordinate oldSize = currentSize;

  currentSize = newCurrentNodeSize;

  int state = ParallelGrid::getParallelCore ()->getNodeState ()[ParallelGrid::getParallelCore ()->getProcessId ()];

  if (state)
  {
    ParallelGridConstructor ();

    /*
     * TODO: do not free/allocate each time
     */
    for (grid_coord index = size.calculateTotalCoord (); index < gridValues.size (); ++index)
    {
      delete gridValues[index];
    }

    grid_coord temp = gridValues.size ();
    gridValues.resize (size.calculateTotalCoord ());

    for (grid_coord index = temp; index < size.calculateTotalCoord (); ++index)
    {
      gridValues[index] = new FieldPointValue ();
    }
  }

  gatherStartPosition ();

  if (state)
  {
    ParallelGridCoordinate chunkStart = getChunkStartPosition ();
    ParallelGridCoordinate chunkEnd = chunkStart + getCurrentSize ();

#ifdef GRID_1D
    ParallelGridCoordinate sizeCoord (chunkEnd.get1 () - chunkStart.get1 () COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
    ParallelGridCoordinate sizeCoord (chunkEnd.get1 () - chunkStart.get1 (),
                                      chunkEnd.get2 () - chunkStart.get2 () COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
    ParallelGridCoordinate sizeCoord (chunkEnd.get1 () - chunkStart.get1 (),
                                      chunkEnd.get2 () - chunkStart.get2 (),
                                      chunkEnd.get3 () - chunkStart.get3 () COORD_TYPES);
#endif /* GRID_3D */

    grid_coord left_coord, right_coord;
    grid_coord down_coord, up_coord;
    grid_coord back_coord, front_coord;

    initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

    grid_coord index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord i = left_coord; i < left_coord + sizeCoord.get1 (); ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = down_coord; j < down_coord + sizeCoord.get2 (); ++j)
      {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        for (grid_coord k = back_coord; k < back_coord + sizeCoord.get3 (); ++k)
        {
#endif /* GRID_3D */

#ifdef GRID_1D
          ParallelGridCoordinate pos (i COORD_TYPES);
          ParallelGridCoordinate posTotal (i - left_coord + chunkStart.get1 () COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
          ParallelGridCoordinate pos (i, j COORD_TYPES);
          ParallelGridCoordinate posTotal (i - left_coord + chunkStart.get1 (),
                                           j - down_coord + chunkStart.get2 () COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
          ParallelGridCoordinate posTotal (i - left_coord + chunkStart.get1 (),
                                           j - down_coord + chunkStart.get2 (),
                                           k - back_coord + chunkStart.get3 () COORD_TYPES);
#endif /* GRID_3D */

          grid_coord coord = calculateIndexFromPosition (pos);

          FieldPointValue *val = gridValues[coord];

          grid_coord coord_total = calculateIndexFromPosition (posTotal, totalSize);

          *val = *totalGrid->getFieldPointValue (coord_total);

#if defined (GRID_3D)
        }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_1D || GRID_2D || GRID_3D */
  }

  delete totalGrid;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalanced grid '%s' for proc: %d (of %d) from raw size: " COORD_MOD ", to raw size " COORD_MOD ". Done\n",
           gridName.data (),
           parallelGridCore->getProcessId (),
           parallelGridCore->getTotalProcCount (),
           oldSize.calculateTotalCoord (),
           currentSize.calculateTotalCoord ());

  /*
   * TODO: check if this share is needed (looks like it's used only for buffers initialiation, which can be done without share)
   * TODO: optimize to fill buffers without sharing
   */
  share ();
} /* ParallelGrid::Resize */

#endif /* DYNAMIC_GRID */

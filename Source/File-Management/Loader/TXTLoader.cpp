#include <iostream>

#include "TXTLoader.h"

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate1D>::skipIndexes (GridCoordinate1D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate2D>::skipIndexes (GridCoordinate2D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get2 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate3D>::skipIndexes (GridCoordinate3D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get2 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get3 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#endif /* MODE_DIM3 */

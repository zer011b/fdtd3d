#include <iomanip>
#include <limits>

#include "TXTDumper.h"

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate1D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate1D & pos) /**< coordinate */
{
  file << pos.get1 () << " ";
} /* TXTDumper::printLine */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate2D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate2D & pos) /**< coordinate */
{
  file << pos.get1 () << " " << pos.get2 () << " ";
} /* TXTDumper::printLine */

#endif /* MODE_DIM2) || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate3D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate3D & pos) /**< coordinate */
{
  file << pos.get1 () << " " << pos.get2 () << " " << pos.get3 () << " ";
} /* TXTDumper::printLine */

#endif /* MODE_DIM3 */

#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "BMPHelper.h"
#include "Dumper.h"

/**
 * Grid saver to BMP files.
 * Template class with coordinate parameter.
 *
 * NOTE: ".bmp" dumper/loader can't reproduce field values precisely.
 *       Consequent dump to and load from ".bmp" file of grid will change grid values.
 */
/*
 * TODO: get rid of duplicate code for dump of 3d, 2d and 1d grids
 */
template <class TCoord>
class BMPDumper: public Dumper<TCoord>
{
  // Helper class for usage with BMP files.
  static BMPHelper BMPhelper;

private:

  // Save grid to file for specific layer.
  void writeToFile (Grid<TCoord> *grid, TCoord, TCoord, int);

  void dumpGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);

  void setupNames (std::string &, std::string &, std::string &,
                   std::string &, std::string &, std::string &, int, int) const;

public:

  virtual ~BMPDumper () {}

  // Virtual method for grid loading.
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;

  void initializeHelper (PaletteType colorPalette, OrthogonalAxis orthAxis)
  {
    BMPhelper.initialize (colorPalette, orthAxis);
  }
};

template <class TCoord>
void
BMPDumper<TCoord>::setupNames (std::string &real, std::string &realtxt,
                               std::string &imag, std::string &imagtxt,
                               std::string &mod, std::string &modtxt, int time_step_back, int coord) const
{
  ASSERT (GridFileManager::names[time_step_back].substr (GridFileManager::names[time_step_back].size () - 4, 4) == std::string (".bmp"));

  real = GridFileManager::names[time_step_back];
  real.resize (real.size () - 4);
  real += std::string ("_[real]");

  imag = GridFileManager::names[time_step_back];
  imag.resize (imag.size () - 4);
  imag += std::string ("_[imag]");

  mod = GridFileManager::names[time_step_back];
  mod.resize (mod.size () - 4);
  mod += std::string ("_[mod]");

  if (coord >= 0)
  {
    real += std::string ("_[coord=") + int64_to_string (coord) + std::string ("]");
    imag += std::string ("_[coord=") + int64_to_string (coord) + std::string ("]");
    mod += std::string ("_[coord=") + int64_to_string (coord) + std::string ("]");
  }

  realtxt = real + std::string (".txt");
  imagtxt = imag + std::string (".txt");
  modtxt = mod + std::string (".txt");

  real += std::string (".bmp");
  imag += std::string (".bmp");
  mod += std::string (".bmp");
}

/**
 * ======== Template implementation ========
 */

template <class TCoord>
void
BMPDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                                     time_step timeStep, int time_step_back)
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Saving grid '" << grid->getName () << "' to BMP image. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Saving grid '" << grid->getName () << "' to BMP image. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  if (time_step_back == -1)
  {
    /**
     * Save all time steps
     */
    for (int i = 0; i < grid->getCountStoredSteps (); ++i)
    {
      writeToFile (grid, startCoord, endCoord, i);
    }
  }
  else
  {
    writeToFile (grid, startCoord, endCoord, time_step_back);
  }

  std::cout << "Saved. " << std::endl;
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
BMPDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back)
{
  int pid = 0;

#ifdef PARALLEL_GRID
  if (SOLVER_SETTINGS.getDoUseParallelGrid ())
  {
    pid = ParallelGrid::getParallelCore ()->getProcessId ();
  }
#endif /* PARALLEL_GRID */

  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_BMP);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

/**
 * Virtual method for grid saving.
 */
template <class TCoord>
void
BMPDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, TCoord startCoord, TCoord endCoord,
                             time_step timeStep, int time_step_back,
                             const std::vector< std::string > & customNames)
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
}

template <class TCoord>
BMPHelper BMPDumper<TCoord>::BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED, OrthogonalAxis::Z);

#endif /* BMP_DUMPER_H */

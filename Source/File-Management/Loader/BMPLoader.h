#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include "BMPHelper.h"
#include "Loader.h"

/**
 * Grid loader from BMP files.
 * Template class with coordinate parameter.
 *
 * NOTE: ".bmp" dumper/loader can't reproduce field values precisely.
 *       Consequent dump to and load from ".bmp" file of grid will change grid values.
 */
template <class TCoord>
class BMPLoader: public Loader<TCoord>
{
  /**
   * Helper class for usage with BMP files.
   */
  static BMPHelper BMPhelper;

private:

  void loadFromFile (Grid<TCoord> *, TCoord, TCoord, int);
  void loadGridInternal (Grid<TCoord> *, TCoord, TCoord, time_step, int);
  void setupNames (std::string &, std::string &, std::string &,
                   std::string &, std::string &, std::string &, int, int) const;
  void loadPixel (grid_coord, grid_coord, Grid<TCoord> *, TCoord, time_step,
                  FPValue, FPValue, FPValue, FPValue,
                  const BMP &, const BMP &);
  void loadTxtFromFile (FPValue, FPValue, FPValue, FPValue,
                        const std::string &, const std::string &);

public:

  virtual ~BMPLoader () {}

  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void loadGrid (Grid<TCoord> *, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;

  /**
   * Initialize color palette and orthogonal axis
   */
  void initializeHelper (PaletteType colorPalette, /**< color palette */
                         OrthogonalAxis orthAxis) /**< orthogonal axis */
  {
    BMPhelper.initialize (colorPalette, orthAxis);
  } /* initializeHelper */
};

/**
 * Load text files from disk
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadTxtFromFile (FPValue maxRe, /**< maximum real value */
                                    FPValue minRe, /**< minimum real value */
                                    FPValue maxIm, /**< maximum imag value */
                                    FPValue minIm, /**< minimum imag value */
                                    const std::string &imageReNameTxt, /**< real text file name */
                                    const std::string &imageImNameTxt) /**< imag text file name */
{
  std::ifstream fileMaxRe;
  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe >> std::setprecision(std::numeric_limits<double>::digits10) >> maxRe >> minRe;
  fileMaxRe.close();
  ASSERT (maxRe >= minRe);

#ifdef COMPLEX_FIELD_VALUES
  std::ifstream fileMaxIm;
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::in);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm >> std::setprecision(std::numeric_limits<double>::digits10) >> maxIm >> minIm;
  fileMaxIm.close();
  ASSERT (maxIm >= minIm);
#endif /* COMPLEX_FIELD_VALUES */
} /* BMPLoader::loadTxtFromFile */

/**
 * Load one pixel from images
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadPixel (grid_coord px, /**< pixel x position */
                              grid_coord py, /**< pixel y position */
                              Grid<TCoord> *grid, /**< grid to load */
                              TCoord pos, /**< position in grid */
                              time_step time_step_back, /**< relative time step at which to load */
                              FPValue minRe, /**< minimum real value */
                              FPValue maxRe, /**< maximum real value */
                              FPValue minIm, /**< minimum imag value */
                              FPValue maxIm, /**< maximum imag value */
                              const BMP &imageRe, /**< real BMP image */
                              const BMP &imageIm) /**< imag BMP image */
{
  RGBApixel pixelRe = imageRe.GetPixel(px, py);
  FPValue currentValRe = BMPhelper.getValueFromPixel (pixelRe, minRe, maxRe);

#ifdef COMPLEX_FIELD_VALUES
  RGBApixel pixelIm = imageIm.GetPixel(px, py);
  FPValue currentValIm = BMPhelper.getValueFromPixel (pixelIm, minIm, maxIm);
#endif /* COMPLEX_FIELD_VALUES */

#ifdef COMPLEX_FIELD_VALUES
  grid->setFieldValue (FieldValue (currentValRe, currentValIm), pos, time_step_back);
#else /* COMPLEX_FIELD_VALUES */
  grid->setFieldValue (FieldValue (currentValRe), pos, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */
} /* BMPLoader::loadPixel */

/**
 * Setup name for files
 */
template <class TCoord>
void
BMPLoader<TCoord>::setupNames (std::string &real, /**< out: name of real image file */
                               std::string &realtxt, /**< out: name of real txt file */
                               std::string &imag, /**< out: name of imag image file */
                               std::string &imagtxt, /**< out: name of imag txt file */
                               std::string &mod, /**< out: name of module image file */
                               std::string &modtxt, /**< out: name of module txt file */
                               int time_step_back, /**< relative time step at which to load */
                               int coord) const /**< coordinate to add to name */
{
  ASSERT (this->GridFileManager::names[time_step_back].substr (this->GridFileManager::names[time_step_back].size () - 4, 4) == std::string (".bmp"));

  real = this->GridFileManager::names[time_step_back];
  real.resize (real.size () - 4);
  real += std::string ("_[real]");

  imag = this->GridFileManager::names[time_step_back];
  imag.resize (imag.size () - 4);
  imag += std::string ("_[imag]");

  mod = this->GridFileManager::names[time_step_back];
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
} /* BMPLoader::setupNames */

/**
 * Choose scenario of loading of grid
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadGridInternal (Grid<TCoord> *grid, /**< grid to load */
                                     TCoord startCoord, /**< start loading from this coordinate */
                                     TCoord endCoord, /**< end loading at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to load */
                                     int time_step_back) /**< relative time step at which to load */
{
  const TCoord& size = grid->getSize ();

  if (time_step_back == -1)
  {
    std::cout << "Loading grid '" << grid->getName () << "' from BMP image. Time step: all"
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }
  else
  {
    std::cout << "Loading grid '" << grid->getName () << "' from BMP image. Time step: " << time_step_back
              << ". Size: " << size.calculateTotalCoord () << " (from startCoord to endCoord). " << std::endl;
  }

  if (time_step_back == -1)
  {
    /**
     * Save all time steps
     */
    for (int i = 0; i < grid->getCountStoredSteps (); ++i)
    {
      loadFromFile (grid, startCoord, endCoord, i);
    }
  }
  else
  {
    loadFromFile (grid, startCoord, endCoord, time_step_back);
  }

  std::cout << "Loaded. " << std::endl;
} /* BMPLoader::loadGridInternal */

/**
 * Virtual method for grid loading, which makes file names automatically
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             int pid) /**< pid of process, which does loading */
{
  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_BMP);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* BMPLoader::loadGrid */

/**
 * Virtual method for grid loading, which uses custom names
 */
template <class TCoord>
void
BMPLoader<TCoord>::loadGrid (Grid<TCoord> *grid, /**< grid to load */
                             TCoord startCoord, /**< start loading from this coordinate */
                             TCoord endCoord, /**< end loading at this coordinate */
                             time_step timeStep, /**< absolute time step at which to load */
                             int time_step_back, /**< relative time step at which to load */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  loadGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* BMPLoader::loadGrid */

template <class TCoord>
BMPHelper BMPLoader<TCoord>::BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED, OrthogonalAxis::Z);

#endif /* BMP_LOADER_H */

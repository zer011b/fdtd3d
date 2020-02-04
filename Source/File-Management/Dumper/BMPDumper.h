#ifndef BMP_DUMPER_H
#define BMP_DUMPER_H

#include "BMPHelper.h"
#include "Dumper.h"

#include <limits>
#include <fstream>
#include <iomanip>

/**
 * Grid saver to BMP files.
 *
 * NOTE: ".bmp" dumper/loader can't reproduce field values precisely.
 *       Consequent dump to and load from ".bmp" file of grid will change grid values.
 */
template <class TCoord>
class BMPDumper: public Dumper<TCoord>
{
  /**
   * Helper class for usage with BMP files.
   */
  static BMPHelper BMPhelper;

private:

  void writeToFile (Grid<TCoord> *grid, TCoord, TCoord, int);
  void dumpGridInternal (Grid<TCoord> *grid, TCoord, TCoord, time_step, int);
  void setupNames (std::string &, std::string &, std::string &,
                   std::string &, std::string &, std::string &, int, int) const;
  void getMaxValues (FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &,
                     Grid<TCoord> *, TCoord, TCoord, int);
  void setPixel (grid_coord, grid_coord, Grid<TCoord> *, TCoord, time_step,
                 FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                 BMP &, BMP &, BMP &);
  void writeTxtToFile (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                       const std::string &, const std::string &, const std::string &);

public:

  virtual ~BMPDumper () {}

  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int, int) CXX11_OVERRIDE;
  virtual void dumpGrid (Grid<TCoord> *grid, TCoord, TCoord, time_step, int,
                         const std::vector< std::string > &) CXX11_OVERRIDE;

  /**
   * Initialize color palette and orthogonal axis
   */
  void initializeHelper (PaletteType colorPalette, /**< color palette */
                         OrthogonalAxis orthAxis) /**< orthogonal axis */
  {
    BMPhelper.initialize (colorPalette, orthAxis);
  } /* initializeHelper */
}; /* BMPDumper */

/**
 * Write text files to disk
 */
template <class TCoord>
void
BMPDumper<TCoord>::writeTxtToFile (FPValue minRe, /**< minimum real value */
                                   FPValue maxRe, /**< maximum real value */
                                   FPValue minIm, /**< minimum imag value */
                                   FPValue maxIm, /**< maximum imag value */
                                   FPValue minMod, /**< minimum module value */
                                   FPValue maxMod, /**< maximum module value */
                                   const std::string &imageReNameTxt, /**< real text file name */
                                   const std::string &imageImNameTxt, /**< imag text file name */
                                   const std::string &imageModNameTxt) /**< module text file name */
{
  std::ofstream fileMaxRe;
#ifdef COMPLEX_FIELD_VALUES
  std::ofstream fileMaxIm;
  std::ofstream fileMaxMod;
#endif /* COMPLEX_FIELD_VALUES */

  fileMaxRe.open (imageReNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxRe.is_open());
  fileMaxRe << std::setprecision(std::numeric_limits<double>::digits10) << maxRe << " " << minRe;
  fileMaxRe.close();

#ifdef COMPLEX_FIELD_VALUES
  fileMaxIm.open (imageImNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxIm.is_open());
  fileMaxIm << std::setprecision(std::numeric_limits<double>::digits10) << maxIm << " " << minIm;
  fileMaxIm.close();

  fileMaxMod.open (imageModNameTxt.c_str (), std::ios::out);
  ASSERT (fileMaxMod.is_open());
  fileMaxMod << std::setprecision(std::numeric_limits<double>::digits10) << maxMod << " " << minMod;
  fileMaxMod.close();
#endif /* COMPLEX_FIELD_VALUES */
} /* BMPDumper::writeTxtToFile */

/**
 * Set one pixel in images
 */
template <class TCoord>
void
BMPDumper<TCoord>::setPixel (grid_coord px, /**< pixel x position */
                             grid_coord py, /**< pixel y position */
                             Grid<TCoord> *grid, /**< grid to save */
                             TCoord pos, /**< position in grid */
                             time_step time_step_back, /**< relative time step at which to save */
                             FPValue minRe, /**< minimum real value */
                             FPValue maxRe, /**< maximum real value */
                             FPValue minIm, /**< minimum imag value */
                             FPValue maxIm, /**< maximum imag value */
                             FPValue minMod, /**< minimum module value */
                             FPValue maxMod, /**< maximum module value */
                             BMP &imageRe, /**< real BMP image */
                             BMP &imageIm, /**< imag BMP image */
                             BMP &imageMod) /**< module BMP image */
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue valueRe = grid->getFieldValue (pos, time_step_back)->real ();
  FPValue valueIm = grid->getFieldValue (pos, time_step_back)->imag ();
  FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
  FPValue valueRe = *grid->getFieldValue (pos, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

  RGBApixel pixelRe = BMPhelper.getPixelFromValue (valueRe, minRe, maxRe);
#ifdef COMPLEX_FIELD_VALUES
  RGBApixel pixelIm = BMPhelper.getPixelFromValue (valueIm, minIm, maxIm);
  RGBApixel pixelMod = BMPhelper.getPixelFromValue (valueMod, minMod, maxMod);
#endif /* COMPLEX_FIELD_VALUES */

  // Set pixel for current image.
  imageRe.SetPixel(px, py, pixelRe);
#ifdef COMPLEX_FIELD_VALUES
  imageIm.SetPixel(px, py, pixelIm);
  imageMod.SetPixel(px, py, pixelMod);
#endif /* COMPLEX_FIELD_VALUES */
} /* BMPDumper::setPixel */

/**
 * Get maximum values in grid
 */
template <class TCoord>
void
BMPDumper<TCoord>::getMaxValues (FPValue &maxRe, /**< out: maximum real */
                                 FPValue &minRe, /**< out: minimum real */
                                 FPValue &maxIm, /**< out: maximum imag */
                                 FPValue &minIm, /**< out: minimum imag */
                                 FPValue &maxMod, /**< out: maximum module */
                                 FPValue &minMod, /**< out: minimum module */
                                 Grid<TCoord> *grid, /**< grid to save */
                                 TCoord startCoord, /**< start saving from this coordinate */
                                 TCoord endCoord, /**< end saving at this coordinate */
                                 int time_step_back) /**< relative time step at which to save */
{
#ifdef COMPLEX_FIELD_VALUES
  maxRe = grid->getFieldValue (startCoord, time_step_back)->real ();
  minRe = maxRe;

  maxIm = grid->getFieldValue (startCoord, time_step_back)->imag ();
  minIm = maxIm;

  maxMod = sqrt (minRe * minRe + minIm * minIm);
  minMod = maxMod;
#else /* COMPLEX_FIELD_VALUES */
  maxRe = *grid->getFieldValue (startCoord, time_step_back);
  minRe = maxRe;
#endif /* !COMPLEX_FIELD_VALUES */

  typename VectorFieldValues<TCoord>::Iterator iter (startCoord, startCoord, endCoord);
  typename VectorFieldValues<TCoord>::Iterator iter_end = VectorFieldValues<TCoord>::Iterator::getEndIterator (startCoord, endCoord);
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();

#ifdef COMPLEX_FIELD_VALUES
    FPValue valueRe = grid->getFieldValue (pos, time_step_back)->real ();
    FPValue valueIm = grid->getFieldValue (pos, time_step_back)->imag ();
    FPValue valueMod = sqrt (valueRe * valueRe + valueIm * valueIm);
#else /* COMPLEX_FIELD_VALUES */
    FPValue valueRe = *grid->getFieldValue (pos, time_step_back);
#endif /* !COMPLEX_FIELD_VALUES */

    if (valueRe > maxRe)
    {
      maxRe = valueRe;
    }
    if (valueRe < minRe)
    {
      minRe = valueRe;
    }

#ifdef COMPLEX_FIELD_VALUES
    if (valueIm > maxIm)
    {
      maxIm = valueIm;
    }
    if (valueIm < minIm)
    {
      minIm = valueIm;
    }

    if (valueMod > maxMod)
    {
      maxMod = valueMod;
    }
    if (valueMod < minMod)
    {
      minMod = valueMod;
    }
#endif /* COMPLEX_FIELD_VALUES */
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "minRe " FP_MOD ", maxRe " FP_MOD ", diffRe " FP_MOD "\n", minRe, maxRe, maxRe - minRe);
#ifdef COMPLEX_FIELD_VALUES
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "minIm " FP_MOD ", maxIm " FP_MOD ", diffIm " FP_MOD "\n", minIm, maxIm, maxIm - minIm);
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "minMod " FP_MOD ", maxMod " FP_MOD ", diffMod " FP_MOD "\n", minMod, maxMod, maxMod - minMod);
#endif /* COMPLEX_FIELD_VALUES */
} /* BMPDumper::getMaxValues */

/**
 * Setup name for files
 */
template <class TCoord>
void
BMPDumper<TCoord>::setupNames (std::string &real, /**< out: name of real image file */
                               std::string &realtxt, /**< out: name of real txt file */
                               std::string &imag, /**< out: name of imag image file */
                               std::string &imagtxt, /**< out: name of imag txt file */
                               std::string &mod, /**< out: name of module image file */
                               std::string &modtxt, /**< out: name of module txt file */
                               int time_step_back, /**< relative time step at which to save */
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
} /* BMPDumper::setupNames */

/**
 * Choose scenario of saving of grid
 */
template <class TCoord>
void
BMPDumper<TCoord>::dumpGridInternal (Grid<TCoord> *grid, /**< grid to save */
                                     TCoord startCoord, /**< start saving from this coordinate */
                                     TCoord endCoord, /**< end saving at this coordinate */
                                     time_step timeStep, /**< absolute time step at which to save */
                                     int time_step_back) /**< relative time step at which to save */
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
} /* BMPDumper::dumpGridInternal */

/**
 * Virtual method for grid saving, which makes file names automatically
 */
template <class TCoord>
void
BMPDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             int pid) /**< pid of process, which does saving */
{
  GridFileManager::setFileNames (grid->getCountStoredSteps (), timeStep, pid, std::string (grid->getName ()), FILE_TYPE_BMP);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* BMPDumper::dumpGrid */

/**
 * Virtual method for grid saving, which uses custom names
 */
template <class TCoord>
void
BMPDumper<TCoord>::dumpGrid (Grid<TCoord> *grid, /**< grid to save */
                             TCoord startCoord, /**< start saving from this coordinate */
                             TCoord endCoord, /**< end saving at this coordinate */
                             time_step timeStep, /**< absolute time step at which to save */
                             int time_step_back, /**< relative time step at which to save */
                             const std::vector< std::string > & customNames) /**< custom names of files */
{
  GridFileManager::setCustomFileNames (customNames);

  dumpGridInternal (grid, startCoord, endCoord, timeStep, time_step_back);
} /* BMPDumper::dumpGrid */

template <class TCoord>
BMPHelper BMPDumper<TCoord>::BMPhelper (PaletteType::PALETTE_BLUE_GREEN_RED, OrthogonalAxis::Z);

#endif /* BMP_DUMPER_H */

#ifndef SETTINGS_H
#define SETTINGS_H

#include "FieldValue.h"
#include "Assert.h"
#include "PhysicsConst.h"

/**
 * Log levels
 */
ENUM_CLASS (LogLevelType, uint8_t,
  LOG_LEVEL_0 = 0,
  LOG_LEVEL_NONE = LOG_LEVEL_0,
  LOG_LEVEL_1,
  LOG_LEVEL_STAGES = LOG_LEVEL_1,
  LOG_LEVEL_2,
  LOG_LEVEL_STAGES_AND_DUMP = LOG_LEVEL_2,
  LOG_LEVEL_3,
  LOG_LEVEL_FULL = LOG_LEVEL_3
);

/**
 * Settings for solver
 */
class Settings
{
private:

  /**
   * Log level
   */
  LogLevelType logLevel;

  /**
   * Size of calculation area
   */
  grid_coord sizeX;
  grid_coord sizeY;
  grid_coord sizeZ;

  /**
   * Size of pml
   */
  grid_coord pmlSizeX;
  grid_coord pmlSizeY;
  grid_coord pmlSizeZ;

  /**
   * Size of tfsf
   */
  grid_coord tfsfSizeX;
  grid_coord tfsfSizeY;
  grid_coord tfsfSizeZ;

  /**
   * Size of ntff
   */
  grid_coord ntffSizeX;
  grid_coord ntffSizeY;
  grid_coord ntffSizeZ;

  /**
   * Number of time steps
   */
  time_step numTimeSteps;

  /**
   * Number of ampiltude mode time steps
   */
  time_step amplitudeTimeSteps;

  /**
   * Buffer size
   */
  int bufSize;

  /**
   * Number of cuda gpus
   */
  int numCudaGPUs;

  /**
   * Number of dimensions
   */
  int dimension;

  /**
   * Angle teta
   */
  FPValue incidentWaveAngle1;

  /**
   * Angle phi
   */
  FPValue incidentWaveAngle2;

  /**
   * Angle psi
   */
  FPValue incidentWaveAngle3;

  /**
   * Flag whether to save result
   */
  bool doDumpRes;

  /**
   * Flag whether to use double material precision
   */
  bool isDoubleMaterialPrecision;

  /**
   * Flag whether to use tfsf
   */
  bool doUseTFSF;

  /**
   * Flag whether to use ntff
   */
  bool doUseNTFF;

  /**
   * Flag whether to use pml
   */
  bool doUsePML;

  /**
   * Flag whether to use metamaterials
   */
  bool doUseMetamaterials;

  /**
   * Flag whether to use amplitude mode
   */
  bool isAmplitudeMode;

  /**
   * dx
   */
  FPValue dx;

  /**
   * Wave length of source
   */
  FPValue sourceWaveLength;

public:

  /**
   * Default constructor
   */
  Settings ()
    : logLevel (LogLevelType::LOG_LEVEL_NONE)
  , sizeX (100)
  , sizeY (100)
  , sizeZ (100)
  , pmlSizeX (10)
  , pmlSizeY (10)
  , pmlSizeZ (10)
  , tfsfSizeX (20)
  , tfsfSizeY (20)
  , tfsfSizeZ (20)
  , ntffSizeX (15)
  , ntffSizeY (15)
  , ntffSizeZ (15)
  , numTimeSteps (100)
  , amplitudeTimeSteps (0)
  , bufSize (1)
  , numCudaGPUs (0)
  , dimension (3)
  , incidentWaveAngle1 (PhysicsConst::Pi / 2)
  , incidentWaveAngle2 (0)
  , incidentWaveAngle3 (PhysicsConst::Pi / 2)
  , doDumpRes (false)
  , isDoubleMaterialPrecision (false)
  , doUseTFSF (false)
  , doUseNTFF (false)
  , doUsePML (false)
  , doUseMetamaterials (false)
  , isAmplitudeMode (false)
  , dx (0.0005)
  , sourceWaveLength (0.02)
  {
  } /* Settings */

  /**
   * Destructor
   */
  ~Settings ()
  {
  } /* ~Settings */

  void setFromCmd (int, char **);

  /**
   * Get log level
   *
   * @return log level
   */
  LogLevelType getLogLevel () const
  {
    return logLevel;
  } /* Settings::getLogLevel */

  /**
   * Get size by x coordinate
   *
   * @return size by x coordinate
   */
  grid_coord getSizeX () const
  {
    return sizeX;
  } /* Settings::getSizeX */

  /**
   * Get size by y coordinate
   *
   * @return size by y coordinate
   */
  grid_coord getSizeY () const
  {
    return sizeY;
  } /* Settings::getSizeY */

  /**
   * Get size by z coordinate
   *
   * @return size by z coordinate
   */
  grid_coord getSizeZ () const
  {
    return sizeZ;
  } /* Settings::getSizeZ */

  /**
   * Get pml size by x coordinate
   *
   * @return pml size by x coordinate
   */
  grid_coord getPmlSizeX () const
  {
    return pmlSizeX;
  } /* Settings::getPmlSizeX */

  /**
   * Get pml size by y coordinate
   *
   * @return pml size by y coordinate
   */
  grid_coord getPmlSizeY () const
  {
    return pmlSizeY;
  } /* Settings::getPmlSizeY */

  /**
   * Get pml size by z coordinate
   *
   * @return pml size by z coordinate
   */
  grid_coord getPmlSizeZ () const
  {
    return pmlSizeZ;
  } /* Settings::getPmlSizeZ */

  /**
   * Get tfsf size by x coordinate
   *
   * @return tfsf size by x coordinate
   */
  grid_coord getTfsfSizeX () const
  {
    return tfsfSizeX;
  } /* Settings::getTfsfSizeX */

  /**
   * Get tfsf size by y coordinate
   *
   * @return tfsf size by y coordinate
   */
  grid_coord getTfsfSizeY () const
  {
    return tfsfSizeY;
  } /* Settings::getTfsfSizeY */

  /**
   * Get tfsf size by z coordinate
   *
   * @return tfsf size by z coordinate
   */
  grid_coord getTfsfSizeZ () const
  {
    return tfsfSizeZ;
  } /* Settings::getTfsfSizeZ */

  /**
   * Get ntff size by x coordinate
   *
   * @return ntff size by x coordinate
   */
  grid_coord getNtffSizeX () const
  {
    return ntffSizeX;
  } /* Settings::getNtffSizeX */

  /**
   * Get ntff size by y coordinate
   *
   * @return ntff size by y coordinate
   */
  grid_coord getNtffSizeY () const
  {
    return ntffSizeY;
  } /* Settings::getNtffSizeY */

  /**
   * Get ntff size by z coordinate
   *
   * @return ntff size by z coordinate
   */
  grid_coord getNtffSizeZ () const
  {
    return ntffSizeZ;
  } /* Settings::getNtffSizeZ */

  /**
   * Get number of time steps
   *
   * @return number of time steps
   */
  time_step getNumTimeSteps () const
  {
    return numTimeSteps;
  } /* Settings::getNumTimeSteps */

  /**
   * Get number of time steps for amplitude mode
   *
   * @return number of time steps for amplitude mode
   */
  time_step getAmplitudeTimeSteps () const
  {
    return amplitudeTimeSteps;
  } /* Settings::getAmplitudeTimeSteps */

  /**
   * Get size of buffer
   *
   * @return size of buffer
   */
  int getBufSize () const
  {
    return bufSize;
  } /* Settings::getBufSize */

  /**
   * Get number of cuda gpus
   *
   * @return number of cuda gpus
   */
  int getNumCudaGPUs () const
  {
    return numCudaGPUs;
  } /* Settings::getNumCudaGPUs */

  /**
   * Get number of dimensions
   *
   * @return number of dimensions
   */
  int getDimension () const
  {
    return dimension;
  } /* Settings::getDimension */

  /**
   * Get angle teta
   *
   * @return angle teta
   */
  FPValue getIncidentWaveAngle1 () const
  {
    return incidentWaveAngle1;
  } /* Settings::getIncidentWaveAngle1 */

  /**
   * Get angle phi
   *
   * @return angle phi
   */
  FPValue getIncidentWaveAngle2 () const
  {
    return incidentWaveAngle2;
  } /* Settings::getIncidentWaveAngle2 */

  /**
   * Get angle psi
   *
   * @return angle psi
   */
  FPValue getIncidentWaveAngle3 () const
  {
    return incidentWaveAngle3;
  } /* Settings::getIncidentWaveAngle3 */

  /**
   * Get flag whether to dump result
   *
   * @return flag whether to dump result
   */
  bool getDoDumpRes () const
  {
    return doDumpRes;
  } /* Settings::getDoDumpRes */

  /**
   * Get flag whether to use double material precision
   *
   * @return flag whether to use double material precision
   */
  bool getIsDoubleMaterialPrecision () const
  {
    return isDoubleMaterialPrecision;
  } /* Settings::getIsDoubleMaterialPrecision */

  /**
   * Get flag whether to use tfsf
   *
   * @return flag whether to use tfsf
   */
  bool getDoUseTFSF () const
  {
    return doUseTFSF;
  } /* Settings::getDoUseTFSF */

  /**
   * Get flag whether to use ntff
   *
   * @return flag whether to use ntff
   */
  bool getDoUseNTFF () const
  {
    return doUseNTFF;
  } /* Settings::getDoUseNTFF */

  /**
   * Get flag whether to use pml
   *
   * @return flag whether to use pml
   */
  bool getDoUsePML () const
  {
    return doUsePML;
  } /* Settings::getDoUsePML */

  /**
   * Get flag whether to use metamaterials
   *
   * @return flag whether to use metamaterials
   */
  bool getDoUseMetamaterials () const
  {
    return doUseMetamaterials;
  } /* Settings::getDoUseMetamaterials */

  /**
   * Get flag whether to use amplitude mode
   *
   * @return flag whether to use amplitude mode
   */
  bool getIsAmplitudeMode () const
  {
    return isAmplitudeMode;
  } /* Settings::getIsAmplitudeMode */

  /**
   * Get dx
   *
   * @return dx
   */
  FPValue getDx () const
  {
    return dx;
  } /* Settings::getDx */

  /**
   * Get source wavelength
   *
   * @return source wavelength
   */
  FPValue getSourceWavelength () const
  {
    return sourceWaveLength;
  } /* Settings::getSourceWavelength */
}; /* Settings */

#endif /* !SETTINGS_H */

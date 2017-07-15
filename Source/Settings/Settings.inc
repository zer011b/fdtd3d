/*
 * SETTINGS_ELEM_FIELD:
 *
 * List of all command line options:
 *
 *  - Name of option
 *  - Getter name
 *  - Option field type
 *  - Default value
 *  - Option cmd arg
 *  - Description
 */

/*
 * SETTINGS_ELEM_OPTION:
 *
 * List of all command line options:
 *
 *  - Option cmd arg
 *  - Flag, whether has argument
 *  - Type of argument
 *  - Description
 */

/*
 * Calculation mode
 */
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d", "2D mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--3d", "3D mode computations")
SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL(logLevel, getLogLevel, LogLevelType, LOG_LEVEL_NONE, "--log-level", "Log level of fdtd3d (0,1,2,3)")

/*
 * Size of calculation area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(sizeX, getSizeX, grid_coord, 100, "--sizex", "Size of calculation area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(sizeY, getSizeY, grid_coord, 100, "--sizey", "Size of calculation area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(sizeZ, getSizeZ, grid_coord, 100, "--sizez", "Size of calculation area by z coordinate")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size", "Use size of calculation area by x coordinate for y and z coordinates too")

/*
 * Size of PML area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeX, getPMLSizeX, grid_coord, 10, "--pml-sizex", "Size of PML area by x coordinate. PML of this size will be applied to both left and right borders of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeY, getPMLSizeY, grid_coord, 10, "--pml-sizey", "Size of PML area by y coordinate. PML of this size will be applied to both left and right borders of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeZ, getPMLSizeZ, grid_coord, 10, "--pml-sizez", "Size of PML area by z coordinate. PML of this size will be applied to both left and right borders of area by z coordinate")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-pml", "Use size of PML area by x coordinate for y and z coordinates too")

/*
 * Size of tfsf area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeX, getTFSFSizeX, grid_coord, 20, "--tfsf-sizex", "Size of TF/SF scattered area by x coordinate. Border of TF/SF will be placed at this distance from both left and right borders of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeY, getTFSFSizeY, grid_coord, 20, "--tfsf-sizey", "Size of TF/SF scattered area by y coordinate. Border of TF/SF will be placed at this distance from both left and right borders of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeZ, getTFSFSizeZ, grid_coord, 20, "--tfsf-sizez", "Size of TF/SF scattered area by z coordinate. Border of TF/SF will be placed at this distance from both left and right borders of area by z coordinate")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-tfsf", "Use size of TF/SF scattered area by x coordinate for y and z coordinates too")

/*
 * Size of ntff area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeX, getNTFFSizeX, grid_coord, 15, "--ntff-sizex", "Size of NTFF area by x coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeY, getNTFFSizeY, grid_coord, 15, "--ntff-sizey", "Size of NTFF area by y coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeZ, getNTFFSizeZ, grid_coord, 15, "--ntff-sizez", "Size of NTFF area by z coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by z coordinate")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-ntff", "Use size of NTFF area by x coordinate for y and z coordinates too")

/*
 * Time steps
 */
SETTINGS_ELEM_FIELD_TYPE_INT(numTimeSteps, getNumTimeSteps, time_step, 100, "--time-steps", "Number of time steps for which to perform computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numAmplitudeTimeSteps, getNumAmplitudeSteps, time_step, 10, "--amplitude-time-steps", "Number of time steps for which to perform amplitude computations")

/*
 * Incident wave angles
 */
SETTINGS_ELEM_FIELD_TYPE_FLOAT(incidentWaveAngle1, getIncidentWaveAngle1, FPValue, 90.0, "--angle-teta", "Incident wave angle teta (degrees)")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(incidentWaveAngle2, getIncidentWaveAngle2, FPValue, 0.0, "--angle-phi", "Incident wave angle phi (degrees)")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(incidentWaveAngle3, getIncidentWaveAngle3, FPValue, 90.0, "--angle-psi", "Incident wave angle psi (degrees)")

/*
 * Concurrency
 */
SETTINGS_ELEM_FIELD_TYPE_INT(bufferSize, getBufferSize, int, 1, "--buffer-size", "Size of buffer for parallel grid")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaGPUs, getNumCudaGPUs, int, 1, "--num-cuda-gpus", "Number of GPUs to use in computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsX, getNumCudaThreadsX, int, 1, "--num-cuda-threads-x", "Number of GPU threads by x coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsY, getNumCudaThreadsY, int, 1, "--num-cuda-threads-y", "Number of GPU threads by y coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsZ, getNumCudaThreadsZ, int, 1, "--num-cuda-threads-z", "Number of GPU threads by z coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseParallelGrid, getDoUseParallelGrid, bool, false, "--parallel-grid", "Use parallel grid (if fdtd3d is built with it)")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseOptimalVirtualTopology, getDoUseOptimalVirtualTopology, bool, false, "--optimal-topology", "Use optimal topology for parallel grid")
SETTINGS_ELEM_FIELD_TYPE_STRING(fileWithAvailableTopologies, getFileWithAvailableTopologies, std::string, "nofile", "--available-topologies", "File with available topologies for current architecture")

SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeX, getTopologySizeX, int, 1, "--topology-sizex", "Size by x coordinate of virtual topology")
SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeY, getTopologySizeY, int, 1, "--topology-sizey", "Size by y coordinate of virtual topology")
SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeZ, getTopologySizeZ, int, 1, "--topology-sizez", "Size by z coordinate of virtual topology")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-topology", "Use size of topology by x coordinate for y and z coordinates too")

/*
 * Computation mode flags
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseDoubleMaterialPrecision, getDoUseDoubleMaterialPrecision, bool, false, "--use-double-material-precision", "Use double material precision")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseTFSF, getDoUseTFSF, bool, false, "--use-tfsf", "Use TF/SF")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseNTFF, getDoUseNTFF, bool, false, "--use-ntff", "Use NTFF")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePML, getDoUsePML, bool, false, "--use-pml", "Use PML")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseMetamaterials, getDoUseMetamaterials, bool, false, "--use-metamaterials", "Use Metamaterials")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseAmplitudeMode, getDoUseAmplitudeMode, bool, false, "--use-amp-mode", "Use amplitude mode")

/*
 * Physics
 */
SETTINGS_ELEM_FIELD_TYPE_FLOAT(gridStep, getGridStep, FPValue, 0.0005, "--dx", "Grid step (meters)")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(sourceWaveLength, getSourceWaveLength, FPValue, 0.02, "--wavelength", "Wave length of source (meters)")

/*
 * Dump flags
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveRes, getDoSaveRes, bool, false, "--save-res", "Save results to files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveMaterials, getDoSaveMaterials, bool, false, "--save-materials", "Save materials to files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveIntermediateRes, getDoSaveIntermediateRes, bool, false, "--save-interm-res", "Save intermediate results to files")
SETTINGS_ELEM_FIELD_TYPE_INT(intermediateSaveStep, getIntermediateSaveStep, time_step, 100, "--interm-save-step", "Save step for intermediate save")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveScatteredFieldRes, getDoSaveScatteredFieldRes, bool, false, "--save-scattered-field-res", "Save scattered field for result")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveScatteredFieldIntermediate, getDoSaveScatteredFieldIntermediate, bool, false, "--save-scattered--field-interm", "Save scattered field for intermediate")

SETTINGS_ELEM_OPTION_TYPE_STRING("--cmd-from-file", "Load command line from file. Cmd file has the next format:\n"
                                                    "\t\t<cmd with arg>\n"
                                                    "\t\t<value>\n"
                                                    "\t\t<cmd with arg>\n"
                                                    "\t\t<value>\n"
                                                    "\t\t<cmd without arg>\n"
                                                    "\t\t<cmd without arg>")

SETTINGS_ELEM_OPTION_TYPE_STRING("--save-cmd-to-file", "Save command line to file")

/*
 * TODO: add
 * --load-from-file
 * --use-point-source
 * --gather-full-grid
 * --save-Ex-only
 * --save-Ey-only
 * --save-Ez-only
 * --save-Hx-only
 * --save-Hy-only
 * --save-Hz-only
 * --save-as-bmp
 * --save-as-dat
 * --save-as-txt
 * --dumper-pallete
 * --dumper-start-coord
 * --dumper-end-coord
 * --dumper-orth-axis
 */

#undef SETTINGS_ELEM_OPTION_TYPE_NONE
#undef SETTINGS_ELEM_OPTION_TYPE_STRING
#undef SETTINGS_ELEM_FIELD_TYPE_NONE
#undef SETTINGS_ELEM_FIELD_TYPE_INT
#undef SETTINGS_ELEM_FIELD_TYPE_FLOAT
#undef SETTINGS_ELEM_FIELD_TYPE_STRING
#undef SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL

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
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-exhy", "1D ExHy mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-exhz", "1D ExHz mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-eyhx", "1D EyHx mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-eyhz", "1D EyHz mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-ezhx", "1D EzHx mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--1d-ezhy", "1D EzHy mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tex", "2D TEx mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tey", "2D TEy mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tez", "2D TEz mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tmx", "2D TMx mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tmy", "2D TMy mode computations")
SETTINGS_ELEM_OPTION_TYPE_NONE("--2d-tmz", "2D TMz mode computations")
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
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeX, getPMLSizeX, grid_coord, 2, "--pml-sizex", "Size of PML area by x coordinate. PML of this size will be applied to both left and right borders of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeY, getPMLSizeY, grid_coord, 2, "--pml-sizey", "Size of PML area by y coordinate. PML of this size will be applied to both left and right borders of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(pmlSizeZ, getPMLSizeZ, grid_coord, 2, "--pml-sizez", "Size of PML area by z coordinate. PML of this size will be applied to both left and right borders of area by z coordinate")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-pml", "Use size of PML area by x coordinate for y and z coordinates too")

/*
 * Size of tfsf area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeXLeft, getTFSFSizeXLeft, grid_coord, 4, "--tfsf-sizex-left", "Size of TF/SF scattered area by x coordinate (left). Border of TF/SF will be placed at this distance from left border of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeXRight, getTFSFSizeXRight, grid_coord, 4, "--tfsf-sizex-right", "Size of TF/SF scattered area by x coordinate (right). Border of TF/SF will be placed at this distance from right border of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeYLeft, getTFSFSizeYLeft, grid_coord, 4, "--tfsf-sizey-left", "Size of TF/SF scattered area by y coordinate (left). Border of TF/SF will be placed at this distance from left border of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeYRight, getTFSFSizeYRight, grid_coord, 4, "--tfsf-sizey-right", "Size of TF/SF scattered area by y coordinate (right). Border of TF/SF will be placed at this distance from right border of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeZLeft, getTFSFSizeZLeft, grid_coord, 4, "--tfsf-sizez-left", "Size of TF/SF scattered area by z coordinate (left). Border of TF/SF will be placed at this distance from left border of area by z coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(tfsfSizeZRight, getTFSFSizeZRight, grid_coord, 4, "--tfsf-sizez-right", "Size of TF/SF scattered area by z coordinate (right). Border of TF/SF will be placed at this distance from right border of area by z coordinate")

SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-tfsf", "Use size of TF/SF scattered area by x coordinate for y and z coordinates too")

/*
 * Size of ntff area
 */
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeX, getNTFFSizeX, grid_coord, 3, "--ntff-sizex", "Size of NTFF area by x coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by x coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeY, getNTFFSizeY, grid_coord, 3, "--ntff-sizey", "Size of NTFF area by y coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by y coordinate")
SETTINGS_ELEM_FIELD_TYPE_INT(ntffSizeZ, getNTFFSizeZ, grid_coord, 3, "--ntff-sizez", "Size of NTFF area by z coordinate. Border of NTFF will be placed at this distance from both left and right borders of area by z coordinate")
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
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsX, getNumCudaThreadsX, int, 4, "--num-cuda-threads-x", "Number of GPU threads by x coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsY, getNumCudaThreadsY, int, 4, "--num-cuda-threads-y", "Number of GPU threads by y coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_INT(numCudaThreadsZ, getNumCudaThreadsZ, int, 4, "--num-cuda-threads-z", "Number of GPU threads by z coordinate to use in computations")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseParallelGrid, getDoUseParallelGrid, bool, false, "--parallel-grid", "Use parallel grid (if fdtd3d is built with it)")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseManualVirtualTopology, getDoUseManualVirtualTopology, bool, false, "--manual-topology", "Use manual topology for parallel grid")

SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeX, getTopologySizeX, int, 1, "--topology-sizex", "Size by x coordinate of virtual topology")
SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeY, getTopologySizeY, int, 1, "--topology-sizey", "Size by y coordinate of virtual topology")
SETTINGS_ELEM_FIELD_TYPE_INT(topologySizeZ, getTopologySizeZ, int, 1, "--topology-sizez", "Size by z coordinate of virtual topology")
SETTINGS_ELEM_OPTION_TYPE_NONE("--same-size-topology", "Use size of topology by x coordinate for y and z coordinates too")

/*
 * CUDA
 */
SETTINGS_ELEM_FIELD_TYPE_INT(cudaBlocksBufferSize, getCudaBlocksBufferSize, int, 1, "--cuda-buffer-size", "Size of buffer for blocks for cuda grid")

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
 * Test border conditions and initial values
 */
// SETTINGS_ELEM_FIELD_TYPE_NONE(doUseExp1BorderCondition, getDoUseExp1BorderCondition, bool, false, "--use-exp1-border-condition", "Exp 1 border conditions")
// SETTINGS_ELEM_FIELD_TYPE_NONE(doUseExp1StartValues, getDoUseExp1StartValues, bool, false, "--use-exp1-start-values", "Exp 1 start values")

SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom1BorderCondition, getDoUsePolinom1BorderCondition, bool, false, "--use-polinom1-border-condition", "Polinom 1 border conditions")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom1StartValues, getDoUsePolinom1StartValues, bool, false, "--use-polinom1-start-values", "Polinom 1 start values")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom1RightSide, getDoUsePolinom1RightSide, bool, false, "--use-polinom1-right-side", "Polinom 1 right side")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculatePolinom1DiffNorm, getDoCalculatePolinom1DiffNorm, bool, false, "--calc-polinom1-diff-norm", "Calculate test norm of difference with exact solution: polinom1")

SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom2BorderCondition, getDoUsePolinom2BorderCondition, bool, false, "--use-polinom2-border-condition", "Polinom 2 border conditions")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom2StartValues, getDoUsePolinom2StartValues, bool, false, "--use-polinom2-start-values", "Polinom 2 start values")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom2RightSide, getDoUsePolinom2RightSide, bool, false, "--use-polinom2-right-side", "Polinom 2 right side")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculatePolinom2DiffNorm, getDoCalculatePolinom2DiffNorm, bool, false, "--calc-polinom2-diff-norm", "Calculate test norm of difference with exact solution: polinom2")

SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom3BorderCondition, getDoUsePolinom3BorderCondition, bool, false, "--use-polinom3-border-condition", "Polinom 3 border conditions")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom3StartValues, getDoUsePolinom3StartValues, bool, false, "--use-polinom3-start-values", "Polinom 3 start values")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePolinom3RightSide, getDoUsePolinom3RightSide, bool, false, "--use-polinom3-right-side", "Polinom 3 right side")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculatePolinom3DiffNorm, getDoCalculatePolinom3DiffNorm, bool, false, "--calc-polinom3-diff-norm", "Calculate test norm of difference with exact solution: polinom3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doUseSin1BorderCondition, getDoUseSin1BorderCondition, bool, false, "--use-sin1-border-condition", "Sin 1 border conditions")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseSin1StartValues, getDoUseSin1StartValues, bool, false, "--use-sin1-start-values", "Sin 1 start values")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateSin1DiffNorm, getDoCalculateSin1DiffNorm, bool, false, "--calc-sin1-diff-norm", "Calculate test norm of difference with exact solution: Sin1")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1ExHyDiffNorm, getDoCalculateExp1ExHyDiffNorm, bool, false, "--calc-exp1-exhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2ExHyDiffNorm, getDoCalculateExp2ExHyDiffNorm, bool, false, "--calc-exp2-exhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3ExHyDiffNorm, getDoCalculateExp3ExHyDiffNorm, bool, false, "--calc-exp3-exhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1ExHzDiffNorm, getDoCalculateExp1ExHzDiffNorm, bool, false, "--calc-exp1-exhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2ExHzDiffNorm, getDoCalculateExp2ExHzDiffNorm, bool, false, "--calc-exp2-exhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3ExHzDiffNorm, getDoCalculateExp3ExHzDiffNorm, bool, false, "--calc-exp3-exhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1EyHxDiffNorm, getDoCalculateExp1EyHxDiffNorm, bool, false, "--calc-exp1-eyhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2EyHxDiffNorm, getDoCalculateExp2EyHxDiffNorm, bool, false, "--calc-exp2-eyhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3EyHxDiffNorm, getDoCalculateExp3EyHxDiffNorm, bool, false, "--calc-exp3-eyhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1EyHzDiffNorm, getDoCalculateExp1EyHzDiffNorm, bool, false, "--calc-exp1-eyhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2EyHzDiffNorm, getDoCalculateExp2EyHzDiffNorm, bool, false, "--calc-exp2-eyhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3EyHzDiffNorm, getDoCalculateExp3EyHzDiffNorm, bool, false, "--calc-exp3-eyhz-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1EzHxDiffNorm, getDoCalculateExp1EzHxDiffNorm, bool, false, "--calc-exp1-ezhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2EzHxDiffNorm, getDoCalculateExp2EzHxDiffNorm, bool, false, "--calc-exp2-ezhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3EzHxDiffNorm, getDoCalculateExp3EzHxDiffNorm, bool, false, "--calc-exp3-ezhx-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp1EzHyDiffNorm, getDoCalculateExp1EzHyDiffNorm, bool, false, "--calc-exp1-ezhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp1")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp2EzHyDiffNorm, getDoCalculateExp2EzHyDiffNorm, bool, false, "--calc-exp2-ezhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp2")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalculateExp3EzHyDiffNorm, getDoCalculateExp3EzHyDiffNorm, bool, false, "--calc-exp3-ezhy-diff-norm", "[USE IN TEST SUITE ONLY] Calculate test norm of difference with exact solution: Exp3")

SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareStartX, getExactSolutionCompareStartX, grid_coord, 0, "--norm-start-x", "Start of norm calculation area by Ox axis")
SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareStartY, getExactSolutionCompareStartY, grid_coord, 0, "--norm-start-y", "Start of norm calculation area by Oy axis")
SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareStartZ, getExactSolutionCompareStartZ, grid_coord, 0, "--norm-start-z", "Start of norm calculation area by Oz axis")

SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareEndX, getExactSolutionCompareEndX, grid_coord, 0, "--norm-end-x", "End of norm calculation area by Ox axis")
SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareEndY, getExactSolutionCompareEndY, grid_coord, 0, "--norm-end-y", "End of norm calculation area by Oy axis")
SETTINGS_ELEM_FIELD_TYPE_INT(exactSolutionCompareEndZ, getExactSolutionCompareEndZ, grid_coord, 0, "--norm-end-z", "End of norm calculation area by Oz axis")

/*
 * NTFF
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalcReverseNTFF, getDoCalcReverseNTFF, bool, false, "--ntff-reverse", "Calculate NTFF reverse diagram")
SETTINGS_ELEM_FIELD_TYPE_STRING(fileNameNTFF, getFileNameNTFF, std::string, "ntff-res.txt", "--ntff-filename", "File to save ntff result")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveNTFFToStdout, getDoSaveNTFFToStdout, bool, false, "--ntff-to-stdout", "Save NTFF for standard output")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(angleStepNTFF, getAngleStepNTFF, FPValue, 10.0, "--ntff-step-angle", "NTFF angle step")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCalcScatteredNTFF, getDoCalcScatteredNTFF, bool, false, "--ntff-scattered", "Calculate NTFF for scattered fields")
SETTINGS_ELEM_FIELD_TYPE_INT(intermediateNTFFStep, getIntermediateNTFFStep, time_step, 100, "--interm-ntff-step", "Save step for intermediate ntff")

/*
 * Physics
 */
SETTINGS_ELEM_FIELD_TYPE_FLOAT(gridStep, getGridStep, FPValue, 0.0005, "--dx", "Grid step (meters)")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(sourceWaveLength, getSourceWaveLength, FPValue, 0.02, "--wavelength", "Wave length of source (meters)")
SETTINGS_ELEM_FIELD_TYPE_FLOAT(courantNum, getCourantNum, FPValue, 0.5, "--courant-factor", "Courant stability factor")

/*
 * Dump flags
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveRes, getDoSaveRes, bool, false, "--save-res", "Save results to files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveResPerProcess, getDoSaveResPerProcess, bool, false, "--save-res-per-process", "Save results to files for each process separately")
// SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveAllRes, getDoSaveAllRes, bool, false, "--save-all-res", "Save all results to files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveMaterials, getDoSaveMaterials, bool, false, "--save-materials", "Save materials to files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveIntermediateRes, getDoSaveIntermediateRes, bool, false, "--save-interm-res", "Save intermediate results to files")
// SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveAllIntermediateRes, getDoSaveAllIntermediateRes, bool, false, "--save-all-interm-res", "Save all intermediate results to files")
SETTINGS_ELEM_FIELD_TYPE_INT(intermediateSaveStep, getIntermediateSaveStep, time_step, 100, "--interm-save-step", "Save step for intermediate save")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveScatteredFieldRes, getDoSaveScatteredFieldRes, bool, false, "--save-scattered-field-res", "Save scattered field for result")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveScatteredFieldIntermediate, getDoSaveScatteredFieldIntermediate, bool, false, "--save-scattered--field-interm", "Save scattered field for intermediate")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveAsBMP, getDoSaveAsBMP, bool, false, "--save-as-bmp", "Save results to .bmp files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveAsDAT, getDoSaveAsDAT, bool, false, "--save-as-dat", "Save results to .dat files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveAsTXT, getDoSaveAsTXT, bool, false, "--save-as-txt", "Save results to .txt files")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveTFSFEInc, getDoSaveTFSFEInc, bool, false, "--save-tfsf-e-incident", "Save TF/SF EInc")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveTFSFHInc, getDoSaveTFSFHInc, bool, false, "--save-tfsf-h-incident", "Save TF/SF HInc")

SETTINGS_ELEM_FIELD_TYPE_NONE(doUseManualStartEndDumpCoord, getDoUseManualStartEndDumpCoord, bool, false, "--manual-save-coords", "Use manual save start and end coordinates")
SETTINGS_ELEM_FIELD_TYPE_INT(saveStartCoordX, getSaveStartCoordX, int, 0, "--save-start-coord-x", "Start x coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_INT(saveStartCoordY, getSaveStartCoordY, int, 0, "--save-start-coord-y", "Start y coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_INT(saveStartCoordZ, getSaveStartCoordZ, int, 0, "--save-start-coord-z", "Start z coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_INT(saveEndCoordX, getSaveEndCoordX, int, 0, "--save-end-coord-x", "End x coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_INT(saveEndCoordY, getSaveEndCoordY, int, 0, "--save-end-coord-y", "End y coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_INT(saveEndCoordZ, getSaveEndCoordZ, int, 0, "--save-end-coord-z", "End z coordinate to save from")
SETTINGS_ELEM_FIELD_TYPE_NONE(doSaveWithoutPML, getDoSaveWithoutPML, bool, false, "--save-no-pml", "Save without PML")

/*
 * Load flags
 */
SETTINGS_ELEM_FIELD_TYPE_STRING(epsFileName, getEpsFileName, std::string, "", "--load-eps-from-file", "File name to load Eps from")
SETTINGS_ELEM_FIELD_TYPE_STRING(muFileName, getMuFileName, std::string, "", "--load-mu-from-file", "File name to load Mu from")
SETTINGS_ELEM_FIELD_TYPE_STRING(omegaPEFileName, getOmegaPEFileName, std::string, "", "--load-omegape-from-file", "File name to load OmegaPE from")
SETTINGS_ELEM_FIELD_TYPE_STRING(omegaPMFileName, getOmegaPMFileName, std::string, "", "--load-omegapm-from-file", "File name to load OmegaPM from")
SETTINGS_ELEM_FIELD_TYPE_STRING(gammaEFileName, getGammaEFileName, std::string, "", "--load-gammae-from-file", "File name to load GammaE from")
SETTINGS_ELEM_FIELD_TYPE_STRING(gammaMFileName, getGammaMFileName, std::string, "", "--load-gammam-from-file", "File name to load GammaM from")

SETTINGS_ELEM_FIELD_TYPE_NONE(useEpsAllNorm, getUseEpsAllNorm, bool, false, "--eps-normed", "Permittivity of Eps material set to 1/eps0")
SETTINGS_ELEM_FIELD_TYPE_NONE(useMuAllNorm, getUseMuAllNorm, bool, false, "--mu-normed", "Permittivity of Mu material set to 1/mu0")

SETTINGS_ELEM_FIELD_TYPE_INT(sphereAccuracy, getSphereAccuracy, int, 100, "--sphere-accuracy", "Sphere approximation accuracy (number of points per grid step)")

SETTINGS_ELEM_FIELD_TYPE_FLOAT(epsSphere, getEpsSphere, FPValue, 1.0, "--eps-sphere", "Permittivity of Eps material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(epsSphereCenterX, getEpsSphereCenterX, int, 0, "--eps-sphere-center-x", "Center position by x coordinate of Eps material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(epsSphereCenterY, getEpsSphereCenterY, int, 0, "--eps-sphere-center-y", "Center position by y coordinate of Eps material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(epsSphereCenterZ, getEpsSphereCenterZ, int, 0, "--eps-sphere-center-z", "Center position by z coordinate of Eps material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(epsSphereRadius, getEpsSphereRadius, int, 0, "--eps-sphere-radius", "Radius of Eps material sphere")

SETTINGS_ELEM_FIELD_TYPE_FLOAT(muSphere, getMuSphere, FPValue, 1.0, "--mu-sphere", "Permittivity of Mu material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(muSphereCenterX, getMuSphereCenterX, int, 0, "--mu-sphere-center-x", "Center position by x coordinate of Mu material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(muSphereCenterY, getMuSphereCenterY, int, 0, "--mu-sphere-center-y", "Center position by y coordinate of Mu material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(muSphereCenterZ, getMuSphereCenterZ, int, 0, "--mu-sphere-center-z", "Center position by z coordinate of Mu material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(muSphereRadius, getMuSphereRadius, int, 0, "--mu-sphere-radius", "Radius of Mu material sphere")

SETTINGS_ELEM_FIELD_TYPE_FLOAT(omegaPESphere, getOmegaPESphere, FPValue, 0.0, "--omegape-sphere", "Electric plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPESphereCenterX, getOmegaPESphereCenterX, int, 0, "--omegape-sphere-center-x", "Center position by x coordinate of electric plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPESphereCenterY, getOmegaPESphereCenterY, int, 0, "--omegape-sphere-center-y", "Center position by y coordinate of electric plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPESphereCenterZ, getOmegaPESphereCenterZ, int, 0, "--omegape-sphere-center-z", "Center position by z coordinate of electric plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPESphereRadius, getOmegaPESphereRadius, int, 0, "--omegape-sphere-radius", "Radius of electric plasma frequency material sphere")

SETTINGS_ELEM_FIELD_TYPE_FLOAT(omegaPMSphere, getOmegaPMSphere, FPValue, 0.0, "--omegapm-sphere", "Magnetic plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPMSphereCenterX, getOmegaPMSphereCenterX, int, 0, "--omegapm-sphere-center-x", "Center position by x coordinate of magnetic plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPMSphereCenterY, getOmegaPMSphereCenterY, int, 0, "--omegapm-sphere-center-y", "Center position by y coordinate of magnetic plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPMSphereCenterZ, getOmegaPMSphereCenterZ, int, 0, "--omegapm-sphere-center-z", "Center position by z coordinate of magnetic plasma frequency material sphere")
SETTINGS_ELEM_FIELD_TYPE_INT(omegaPMSphereRadius, getOmegaPMSphereRadius, int, 0, "--omegapm-sphere-radius", "Radius of magnetic plasma frequency material sphere")

/*
SETTINGS_ELEM_FIELD_TYPE_STRING(exFileName, getExFileName, std::string, "", "--load-ex-from-file", "File name to load Ex from")
SETTINGS_ELEM_FIELD_TYPE_STRING(eyFileName, getEyFileName, std::string, "", "--load-ey-from-file", "File name to load Ey from")
SETTINGS_ELEM_FIELD_TYPE_STRING(ezFileName, getEzFileName, std::string, "", "--load-ez-from-file", "File name to load Ez from")
SETTINGS_ELEM_FIELD_TYPE_STRING(hxFileName, getHxFileName, std::string, "", "--load-hx-from-file", "File name to load Hx from")
SETTINGS_ELEM_FIELD_TYPE_STRING(hyFileName, getHyFileName, std::string, "", "--load-hy-from-file", "File name to load Hy from")
SETTINGS_ELEM_FIELD_TYPE_STRING(hzFileName, getHzFileName, std::string, "", "--load-hz-from-file", "File name to load Hz from")

SETTINGS_ELEM_FIELD_TYPE_STRING(dxFileName, getDxFileName, std::string, "", "--load-dx-from-file", "File name to load Dx from")
SETTINGS_ELEM_FIELD_TYPE_STRING(dyFileName, getDyFileName, std::string, "", "--load-dy-from-file", "File name to load Dy from")
SETTINGS_ELEM_FIELD_TYPE_STRING(dzFileName, getDzFileName, std::string, "", "--load-dz-from-file", "File name to load Dz from")
SETTINGS_ELEM_FIELD_TYPE_STRING(bxFileName, getbxFileName, std::string, "", "--load-bx-from-file", "File name to load Bx from")
SETTINGS_ELEM_FIELD_TYPE_STRING(byFileName, getbyFileName, std::string, "", "--load-by-from-file", "File name to load By from")
SETTINGS_ELEM_FIELD_TYPE_STRING(bzFileName, getbzFileName, std::string, "", "--load-bz-from-file", "File name to load Bz from")

SETTINGS_ELEM_FIELD_TYPE_STRING(d1xFileName, getD1xFileName, std::string, "", "--load-d1x-from-file", "File name to load D1x from")
SETTINGS_ELEM_FIELD_TYPE_STRING(d1yFileName, getD1yFileName, std::string, "", "--load-d1y-from-file", "File name to load D1y from")
SETTINGS_ELEM_FIELD_TYPE_STRING(d1zFileName, getD1zFileName, std::string, "", "--load-d1z-from-file", "File name to load D1z from")
SETTINGS_ELEM_FIELD_TYPE_STRING(b1xFileName, getB1xFileName, std::string, "", "--load-b1x-from-file", "File name to load B1x from")
SETTINGS_ELEM_FIELD_TYPE_STRING(b1yFileName, getB1yFileName, std::string, "", "--load-b1y-from-file", "File name to load B1y from")
SETTINGS_ELEM_FIELD_TYPE_STRING(b1zFileName, getB1zFileName, std::string, "", "--load-b1z-from-file", "File name to load B1z from")
*/

/*
 * BMP flags
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePaletteGray, getDoUsePaletteGray, bool, false, "--palette-gray", "Use gray palette for .bmp dump")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePaletteRGB, getDoUsePaletteRGB, bool, false, "--palette-rgb", "Use RGB palette for .bmp dump")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseOrthAxisX, getDoUseOrthAxisX, bool, false, "--orth-axis-x", "Use Ox orthogonal axis for .bmp dump")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseOrthAxisY, getDoUseOrthAxisY, bool, false, "--orth-axis-y", "Use Oy orthogonal axis for .bmp dump")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseOrthAxisZ, getDoUseOrthAxisZ, bool, false, "--orth-axis-z", "Use Oz orthogonal axis for .bmp dump")

/*
 * Point source
 */
SETTINGS_ELEM_FIELD_TYPE_INT(pointSourcePositionX, getPointSourcePositionX, grid_coord, 0, "--point-source-pos-x", "Point source position by Ox axis")
SETTINGS_ELEM_FIELD_TYPE_INT(pointSourcePositionY, getPointSourcePositionY, grid_coord, 0, "--point-source-pos-y", "Point source position by Oy axis")
SETTINGS_ELEM_FIELD_TYPE_INT(pointSourcePositionZ, getPointSourcePositionZ, grid_coord, 0, "--point-source-pos-z", "Point source position by Oz axis")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceEx, getDoUsePointSourceEx, bool, false, "--point-source-ex", "Point source Ex")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceEy, getDoUsePointSourceEy, bool, false, "--point-source-ey", "Point source Ey")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceEz, getDoUsePointSourceEz, bool, false, "--point-source-ez", "Point source Ez")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceHx, getDoUsePointSourceHx, bool, false, "--point-source-hx", "Point source Hx")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceHy, getDoUsePointSourceHy, bool, false, "--point-source-hy", "Point source Hy")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUsePointSourceHz, getDoUsePointSourceHz, bool, false, "--point-source-hz", "Point source Hz")

/*
 * Dynamic grid
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseDynamicGrid, getDoUseDynamicGrid, bool, false, "--dynamic-grid", "Use dynamic grid (if fdtd3d is built with it)")
SETTINGS_ELEM_FIELD_TYPE_INT(rebalanceStep, getRebalanceStep, int, 100, "--rebalance-step", "Rebalance step for dynamic parallel grid")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCheckDisablingConditions, getDoCheckDisablingConditions, bool, false, "--check-disabling-cond", "Check dynamic grid disabling conditions")
SETTINGS_ELEM_FIELD_TYPE_NONE(doCheckEnablingConditions, getDoCheckEnablingConditions, bool, false, "--check-enabling-cond", "Check dynamic grid enabling conditions")

/*
 * FDTD helper grids
 */
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseCaCbGrids, getDoUseCaCbGrids, bool, false, "--use-ca-cb", "Use helper grids (Ca, Cb, Da, Db) with precomputed values for general FDTD computation")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseCaCbPMLGrids, getDoUseCaCbPMLGrids, bool, false, "--use-ca-cb-pml", "Use helper grids (Ca, Cb, Cc, Da, Db, Dc) with precomputed values for PML FDTD computation")
SETTINGS_ELEM_FIELD_TYPE_NONE(doUseCaCbPMLMetaGrids, getDoUseCaCbPMLMetaGrids, bool, false, "--use-ca-cb-pml-metamaterials", "Use helper grids (B0, B1, B2, A1, A2) with precomputed values for PML metamaterials FDTD computation")

/*
 * Steps in time
 */
SETTINGS_ELEM_FIELD_TYPE_INT(storedSteps, getStoredSteps, time_step, 2, "--stored-steps", "Number of time steps in time, for which grid values are stored")

SETTINGS_ELEM_OPTION_TYPE_STRING("--cmd-from-file", "Load command line from file. Cmd file has the next format:\n"
                                                    "\t\t<cmd with arg>\n"
                                                    "\t\t<value>\n"
                                                    "\t\t<cmd with arg>\n"
                                                    "\t\t<value>\n"
                                                    "\t\t<cmd without arg>\n"
                                                    "\t\t// <comment>\n"
                                                    "\t\t# <comment>\n"
                                                    "\t\t<cmd without arg>")

SETTINGS_ELEM_OPTION_TYPE_STRING("--save-cmd-to-file", "Save command line to file")

#undef SETTINGS_ELEM_OPTION_TYPE_NONE
#undef SETTINGS_ELEM_OPTION_TYPE_STRING
#undef SETTINGS_ELEM_FIELD_TYPE_NONE
#undef SETTINGS_ELEM_FIELD_TYPE_INT
#undef SETTINGS_ELEM_FIELD_TYPE_FLOAT
#undef SETTINGS_ELEM_FIELD_TYPE_STRING
#undef SETTINGS_ELEM_FIELD_TYPE_LOG_LEVEL

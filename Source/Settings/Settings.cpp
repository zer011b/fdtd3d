#include "Settings.h"

#include <string.h>

#ifdef CXX11_ENABLED
#else
#include <stdlib.h>
#endif

Settings solverSettings;

/**
 * Set settings from command line arguments
 */
void
Settings::setFromCmd (int argc, /**< number of arguments */
                      char **argv) /**< arguments */
{
  for (int i = 1; i < argc; ++i)
  {
    if (strcmp (argv[i], "--help") == 0)
    {
      printf ("fdtd3d is an open source 1D [NYI], 2D, 3D FDTD electromagnetics solver with MPI, OpenMP [NYI] and CUDA support [WIP].\n");
      printf ("Usage: fdtd3d [options]\n\n");
      printf ("Options:\n");
      printf ("  --log-level N (int)\n");
      printf ("\n");
      printf ("  --sizex N (int)\n");
      printf ("  --sizey N (int)\n");
      printf ("  --sizez N (int)\n");
      printf ("  --same-size N (int)\n");
      printf ("\n");
      printf ("  --pml-sizex N (int)\n");
      printf ("  --pml-sizey N (int)\n");
      printf ("  --pml-sizez N (int)\n");
      printf ("  --same-size-pml N (int)\n");
      printf ("\n");
      printf ("  --tfsf-sizex N (int)\n");
      printf ("  --tfsf-sizey N (int)\n");
      printf ("  --tfsf-sizez N (int)\n");
      printf ("  --same-size-tfsf N (int)\n");
      printf ("\n");
      printf ("  --ntff-sizex N (int)\n");
      printf ("  --ntff-sizey N (int)\n");
      printf ("  --ntff-sizez N (int)\n");
      printf ("  --same-size-ntff N (int)\n");
      printf ("\n");
      printf ("  --time-steps N (int)\n");
      printf ("  --amplitude-time-steps N (int)\n");
      printf ("  --buffer-size N (int)\n");
      printf ("  --num-gpu N (int)\n");
      printf ("  --2d\n");
      printf ("  --3d\n");
      printf ("  --angle-teta N (degrees, floating point)\n");
      printf ("  --angle-phi N (degrees, floating point)\n");
      printf ("  --angle-psi N (degrees, floating point)\n");
      printf ("  --save-res\n");
      printf ("  --double-material-precision\n");
      printf ("  --use-tfsf\n");
      printf ("  --use-ntff\n");
      printf ("  --use-pml\n");
      printf ("  --use-metamaterials\n");
      printf ("  --amplitude-mode\n");
      printf ("  --dx\n");
      printf ("  --wavelength\n");

      exit (0);
    }
    else if (strcmp (argv[i], "--log-level") == 0)
    {
      ++i;
      switch (STOI (argv[i]))
      {
        case 0:
        {
          logLevel = LogLevelType::LOG_LEVEL_0;
          break;
        }
        case 1:
        {
          logLevel = LogLevelType::LOG_LEVEL_1;
          break;
        }
        case 2:
        {
          logLevel = LogLevelType::LOG_LEVEL_2;
          break;
        }
        case 3:
        {
          logLevel = LogLevelType::LOG_LEVEL_3;
          break;
        }
        default:
        {
          UNREACHABLE;
        }
      }

    }
    else if (strcmp (argv[i], "--sizex") == 0)
    {
      ++i;
      sizeX = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizey") == 0)
    {
      ++i;
      sizeY = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--sizez") == 0)
    {
      ++i;
      sizeZ = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--same-size") == 0)
    {
      sizeZ = sizeY = sizeX;
    }
    else if (strcmp (argv[i], "--pml-sizex") == 0)
    {
      ++i;
      pmlSizeX = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--pml-sizey") == 0)
    {
      ++i;
      pmlSizeY = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--pml-sizez") == 0)
    {
      ++i;
      pmlSizeZ = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--same-size-pml") == 0)
    {
      pmlSizeZ = pmlSizeY = pmlSizeX;
    }
    else if (strcmp (argv[i], "--tfsf-sizex") == 0)
    {
      ++i;
      tfsfSizeX = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--tfsf-sizey") == 0)
    {
      ++i;
      tfsfSizeY = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--tfsf-sizez") == 0)
    {
      ++i;
      tfsfSizeZ = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--same-size-tfsf") == 0)
    {
      tfsfSizeZ = tfsfSizeY = tfsfSizeX;
    }
    else if (strcmp (argv[i], "--ntff-sizex") == 0)
    {
      ++i;
      ntffSizeX = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--ntff-sizey") == 0)
    {
      ++i;
      ntffSizeY = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--ntff-sizez") == 0)
    {
      ++i;
      ntffSizeZ = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--same-size-ntff") == 0)
    {
      ntffSizeZ = tfsfSizeY = tfsfSizeX;
    }
    else if (strcmp (argv[i], "--time-steps") == 0)
    {
      ++i;
      numTimeSteps = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--amplitude-time-steps") == 0)
    {
      ++i;
      amplitudeTimeSteps = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--buffer-size") == 0)
    {
      ++i;
      bufSize = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--num-gpu") == 0)
    {
      ++i;
      numCudaGPUs = STOI (argv[i]);
    }
    else if (strcmp (argv[i], "--2d") == 0)
    {
      dimension = 2;
    }
    else if (strcmp (argv[i], "--3d") == 0)
    {
      dimension = 3;
    }
    else if (strcmp (argv[i], "--angle-teta") == 0)
    {
      ++i;
      incidentWaveAngle1 = STOF (argv[i]) * PhysicsConst::Pi / 180.0;
    }
    else if (strcmp (argv[i], "--angle-phi") == 0)
    {
      ++i;
      incidentWaveAngle2 = STOF (argv[i]) * PhysicsConst::Pi / 180.0;
    }
    else if (strcmp (argv[i], "--angle-psi") == 0)
    {
      ++i;
      incidentWaveAngle3 = STOF (argv[i]) * PhysicsConst::Pi / 180.0;
    }
    else if (strcmp (argv[i], "--save-res") == 0)
    {
      doDumpRes = true;
    }
    else if (strcmp (argv[i], "--double-material-precision") == 0)
    {
      isDoubleMaterialPrecision = true;
    }
    else if (strcmp (argv[i], "--use-tfsf") == 0)
    {
      doUseTFSF = true;
    }
    else if (strcmp (argv[i], "--use-ntff") == 0)
    {
      doUseNTFF = true;
    }
    else if (strcmp (argv[i], "--use-pml") == 0)
    {
      doUsePML = true;
    }
    else if (strcmp (argv[i], "--use-metamaterials") == 0)
    {
      doUseMetamaterials = true;
    }
    else if (strcmp (argv[i], "--amplitude-mode") == 0)
    {
      isAmplitudeMode = true;
    }
    else if (strcmp (argv[i], "--dx") == 0)
    {
      ++i;
      dx = STOF (argv[i]);
    }
    else if (strcmp (argv[i], "--wavelength") == 0)
    {
      ++i;
      sourceWaveLength = STOF (argv[i]);
    }
    else
    {
      printf ("Unknown option [%s]\n", argv[i]);
    }
  }
} /* Settings::setFromCmd */

# fdtd3d
3D FDTD solver.

# Build Process
Build is done using cmake & make.
```sh
mkdir build
cd build
cmake ..
make
```

# Build Flags
```c_cpp
-DCMAKE_BUILD_TYPE - build type (Debug or Release)
-DFULL_VALUES - use double (ON) or float (OFF)
-DTIME_STEPS - number of steps in time (1 or 2)
-DGRID_DIMENSION - number of dimensions in grid (1, 2 or 3)
```

# Preprocessor variables
```c_cpp
FULL_VALUES - use double (TRUE) or float (FALSE)
ONE_TIME_STEP - one previous time step is saved
TWO_TIME_STEPS - two previous time steps are saved
GRID_1D - one-dimensional solver
GRID_2D - two-dimensional solver
GRID_3D - three-dimensional solver
```

# About
This solver is an MPI/OpenMP implementation of fdtd solver. Solver currently supports only 3D grids. Solver is manually set to perform calculations of one specific problem and needs to be manually modified to perform other calculations. Current code structure is not the best and will be modified in future. Performance will be tested and improved as well.

EasyBMP lib is used to output resulting electromagnetic fields. It is downloaded from sourceforge and used as is.

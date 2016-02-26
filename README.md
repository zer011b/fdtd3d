[![Build](https://img.shields.io/travis/rust-lang/rust.svg)](https://github.com/zer011b/fdtd3d)
[![Parallel grid](https://img.shields.io/badge/Parallel%20Grid-1D%20and%202D-blue.svg)](https://github.com/zer011b/fdtd3d)
[![FDTD mode](https://img.shields.io/badge/FDTD-Ez%20mode%20only-red.svg)](https://github.com/zer011b/fdtd3d)

# fdtd3d

This is an open-source implementation of FDTD Maxwell's equations solver for different dimensions (1, 2 or 3) with support of concurrency (MPI/OpenMP) if required. The key idea is building of solver for your specific needs with different components, i.e. concurrency support, parallel buffer types, specific dimension and others.  

For additional info on current project development status and future plans check issues and milestones.

# Build Process

Build is done using cmake.

## Release Build

```sh
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Debug Build

```sh
mkdir Debug
cd Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```
## Additional Example

main.cpp has example of parallel grid usage, independent of dimension. Build command for 3D grid:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DFULL_VALUES=ON -DTIME_STEPS=2 -DGRID_DIMENSION=3 -DPRINT_MESSAGE=OFF -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=xyz
```

# Build Flags
```c_cpp
-DCMAKE_BUILD_TYPE - build type (Debug or Release)
-DFULL_VALUES - use double (ON) or float (OFF)
-DTIME_STEPS - number of steps in time (1 or 2)
-DGRID_DIMENSION - number of dimensions in grid (1, 2 or 3)
-DPRINT_MESSAGE - print output (ON or OFF)
-DPARALLEL_GRID - use parallel grid or not (ON or OFF)
-DPARALLEL_BUFFER_DIMENSION - dimension of parallel buffers, i.e. actual coordinate systems (x, y, z, xy, yz, xz, xyz)
```

# Preprocessor variables
```c_cpp
FULL_VALUES - use double (TRUE) or float (FALSE)
ONE_TIME_STEP - one previous time step is saved
TWO_TIME_STEPS - two previous time steps are saved
GRID_1D - one-dimensional solver
GRID_2D - two-dimensional solver
GRID_3D - three-dimensional solver
PRINT_MESSAGE - print output (TRUE of FALSE)
PARALLEL_GRID - use parallel grid or not (TRUE or FALSE)
PARALLEL_BUFFER_DIMENSION_1D_X - one dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_1D_Y - one dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_1D_Z - one dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_2D_XY - two dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_2D_YZ - two dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_2D_XZ - two dimensional parallel buffer
PARALLEL_BUFFER_DIMENSION_3D_XYZ - three dimensional parallel buffer
```

# About

Feel free to ask any questions.

EasyBMP lib is used to output resulting electromagnetic fields. It is downloaded from sourceforge and used as is.

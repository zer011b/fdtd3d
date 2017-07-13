[![Build Status](https://travis-ci.org/zer011b/fdtd3d.svg?branch=master)](https://travis-ci.org/zer011b/fdtd3d)

# fdtd3d

This is an open-source implementation of FDTD Maxwell's equations solver for different dimensions (1, 2 or 3) with support of concurrency (MPI/OpenMP/Cuda) if required. The key idea is building of solver for your specific needs with different components, i.e. concurrency support with enabled MPI, OpenMP or GPU support, parallel buffer types, specific dimension and others.

For additional info on current project development status and future plans check issues and milestones.

# Build Process

Build is done using cmake. Also there are build and run scripts you could find useful.

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

Build command for 3D grid:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DTIME_STEPS=2 -DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=OFF -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=xyz -DCXX11_ENABLED=ON -DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50
```

# Testing

To start unit tests do next

```sh
./Tools/build-and-run-unit-tests.sh <home_dir_of_the_project> <cxx_compiler> <c_compiler> false
```

# Build Flags

Solver incorporates following features which could be set up during build.

```c_cpp
CMAKE_BUILD_TYPE - build type (Debug or Release)
VALUE_TYPE - use float (f), double (d) or long double (ld) floating point values
COMPLEX_FIELD_VALUES - use complex values or not (ON of OFF)
TIME_STEPS - number of steps in time (1 or 2)
PARALLEL_GRID_DIMENSION - number of dimensions in parallel grid (1, 2 or 3)
PRINT_MESSAGE - print debug output (ON or OFF)
PARALLEL_GRID - use parallel grid or not (ON or OFF)
PARALLEL_BUFFER_DIMENSION - dimension of parallel buffers, i.e. actual coordinate systems (x, y, z, xy, yz, xz, xyz)
CXX11_ENABLED - allow support of C++11 (ON or OFF)
CUDA_ENABLED - enable support of GPU (ON or OFF)
CUDA_ARCH_SM_TYPE - sm type for GPU
```

If any of the flags change or some new are added, testing scripts should be updated.

# Launch

```sh
cd Release/Source
# show help
./fdtd3d --help

# show version of solver
./fdtd3d --version

# example of launch command for 3D build
./fdtd3d --save-res --time-steps 10 --sizex 80 --same-size --use-tfsf --3d --angle-phi 0 --use-pml --dx 0.0005 --wavelength 0.02

# example of launch with command line file
./fdtd3d --cmd-from-file cmd.txt

# with cmd.txt file having the next format
#
# --save-res
# --time-steps
# 1
# --sizex
# 80
# --same-size
# --use-tfsf
# --3d
# --angle-phi
# 0
# --use-pml
# --dx
# 0.0005
# --wavelength
# 0.02
```

# About

Feel free to ask any questions.

EasyBMP lib is used to output resulting electromagnetic fields. It is downloaded from sourceforge and used as is.

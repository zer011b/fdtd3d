[![Build Status](https://travis-ci.org/zer011b/fdtd3d.svg?branch=master)](https://travis-ci.org/zer011b/fdtd3d)

# fdtd3d

This is an open-source implementation of FDTD Maxwell's equations solver for different dimensions (1, 2 or 3) with support of concurrency (MPI/OpenMP/Cuda) if required. The key idea is building of solver for your specific needs with different components, i.e. concurrency support with enabled MPI, OpenMP or GPU support, parallel buffer types, specific dimension and others.

For additional info on current project development status and future plans check issues and milestones.

# Build Process

Build is done using cmake. Also there are build and run scripts you could find useful.

## Prerequisites

Cuda build requires `cmake >= 3.8`. Cuda builds with older versions of cmake are not supported. To manually build `cmake` run next commands:

```sh
./install-cmake.sh
export PATH=`pwd`/Third-party/cmake/bin:$PATH
```

Non-cuda builds support `cmake >= 3.0.2`, but `CMakeLists.txt` will still require 3.8 version. To build with older versions run next commands:
```sh
sed -i 's/cmake_minimum_required(VERSION 3\.8)/cmake_minimum_required(VERSION 3\.0\.2)/' CMakeLists.txt
```

## Build

```sh
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Additional Example

Build command for 3D grid in detail:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=OFF -DTIME_STEPS=2 \
-DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=ON -DPARALLEL_GRID=OFF -DPARALLEL_BUFFER_DIMENSION=xyz \
-DCXX11_ENABLED=ON -DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50 -DLARGE_COORDINATES=OFF
```

# Testing

Testing is performed using Travis CI. Open pull request with your changes and it will be automatically tested.
For details, see .travis.yml.

# Build Flags

Solver incorporates following features which could be set up during build.

```c_cpp
CMAKE_BUILD_TYPE - build type (Debug, ReleaseWithAsserts, Release)
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
LARGE_COORDINATES - whether to use int64 for grid coordinates or int32 (ON or OFF)
STD_COMPEX - use std::complex instead of custom CComplex class (std::complex is not supported with Cuda)
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
./fdtd3d --load-eps-from-file /tmp/eps.txt --save-res --time-steps 10 --sizex 80 --same-size --use-tfsf \
         --3d --angle-phi 0 --use-pml --dx 0.0005 --wavelength 0.02 --save-cmd-to-file cmd.txt

# example of the same launch with command line file
./fdtd3d --cmd-from-file cmd.txt

# cmd.txt file has the next format
#
# --load-eps-from-file
# /tmp/eps.txt
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

You can find some examples in `./Examples`. See [Input & Output](Docs/Input-Output.md) for details about load and save of files.

# Documentation

Doxygen documentation is available at [Documentation](http://zer011b.github.io/fdtd3d/).

To generate it manually from config in `./Doxyfile` run next commands:

```sh
sudo apt-get install doxygen
doxygen
firefox docs/index.html
```

# How to cite

You can site the following paper about the techniques used in fdtd3d:

Balykov G. (2017) Parallel FDTD Solver with Optimal Topology and Dynamic Balancing. In: Voevodin V., Sobolev S. (eds) Supercomputing. RuSCDays 2017. Communications in Computer and Information Science, vol 793. Springer, Cham

# About

Feel free to ask any questions.

EasyBMP lib is used to output resulting electromagnetic fields. It is downloaded from sourceforge and used as is.

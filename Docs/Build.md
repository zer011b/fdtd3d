# Prerequisites

Cuda build requires `cmake >= 3.8`. Cuda builds with older versions of cmake are not supported. To manually build `cmake` run next commands:

```sh
./install-cmake.sh
export PATH=`pwd`/Third-party/cmake/bin:$PATH
```

Non-cuda builds support `cmake >= 3.0.2`, but `CMakeLists.txt` will still require 3.8 version. To build with older versions run next commands:
```sh
sed -i 's/cmake_minimum_required(VERSION 3\.8)/cmake_minimum_required(VERSION 3\.0\.2)/' CMakeLists.txt
```

# Build Flags

Solver incorporates following features which could be set up during build.

```c_cpp
CMAKE_BUILD_TYPE - build type (Debug, RelWithDebInfo, Release)
SOLVER_DIM_MODES - dimension modes to include in build (EX_HY;EX_HZ;EY_HX;EY_HZ;EZ_HX;EZ_HY;TEX;TEY;TEZ;TMX;TMY;TMZ;DIM1;DIM2;DIM3;ALL); default value is ALL, which includes all the supported modes
VALUE_TYPE - use float (f), double (d) or long double (ld) floating point values
COMPLEX_FIELD_VALUES - use complex values or not (ON of OFF)
PARALLEL_GRID_DIMENSION - number of dimensions in parallel grid (1, 2 or 3)
PRINT_MESSAGE - print debug output (ON or OFF)
PARALLEL_GRID - use parallel grid or not (ON or OFF)
PARALLEL_BUFFER_DIMENSION - dimension of parallel buffers, i.e. actual coordinate systems (x, y, z, xy, yz, xz, xyz)
CUDA_ENABLED - enable support of GPU (ON or OFF)
CUDA_ARCH_SM_TYPE - sm type for GPU
LARGE_COORDINATES - whether to use int64 for grid coordinates or int32 (ON or OFF)
STD_COMPLEX - use std::complex instead of custom CComplex class (std::complex is not supported with Cuda)
```

If any of the flags change or some new are added, testing scripts should be updated.


# Additional Example

Build command for 3D grid in detail:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOLVER_DIM_MODES=DIM3 -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=OFF \
-DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=ON -DPARALLEL_GRID=OFF -DPARALLEL_BUFFER_DIMENSION=xyz \
-DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50 -DLARGE_COORDINATES=OFF
```

# MPI+Cuda Build

To build parallel mode with GPU support (MPI+Cuda) do not use mpicc and mpicxx compiler wrappers, because they do not set `MPI_CXX_INCLUDE_DIRS` and `MPI_CXX_LIBRARIES`. These are required to be passed to `nvcc`.

# Build for BlueGene\P

BlueGene\P doesn't have cmake provided, so it should be built from source. Download cmake source and then do the following

```sh
./bootstrap
make

# make install won't work because of lack of root access for ordinary users
```

GCC provided for BlueGene\P is heavily outdated (4.1.2), thus, it doesn't support c++11 features. Also, apply next patch to fdtd3d:

```patch
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 2dc58ad..838b491 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1,4 +1,6 @@
-cmake_minimum_required(VERSION 3.8)
+#cmake_minimum_required(VERSION 3.8)
+cmake_policy(SET CMP0057 NEW)
+cmake_policy(SET CMP0012 NEW)

 project (fdtd3d LANGUAGES CXX)

```

Build command:
```sh
../../cmake/cmake-3.6.0-rc1/bin/cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DSOLVER_DIM_MODES=DIM3 -DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=ON -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=xyz -DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50 -DDYNAMIC_GRID=OFF -DCOMBINED_SENDRECV=ON -DMPI_CLOCK=OFF -DCMAKE_C_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicc -DCMAKE_CXX_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicxx
make fdtd3d
```

## Issues

### IBM XL compilers undefined reference

mpixlcxx and mpixlcxx_r compilers lead to undefined references during linking (supposed that this is a bug in IBM XL compilers). GCC should be used instead (`/bgsys/drivers/ppcfloor/comm/bin/mpicxx`):

```sh
../../../../bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DSOLVER_DIM_MODES=DIM3 -DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=OFF -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=x -DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50 -DDYNAMIC_GRID=OFF -DCOMBINED_SENDRECV=ON -DMPI_CLOCK=OFF -DCMAKE_C_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicc -DCMAKE_CXX_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicxx
```

### cmake linking error

Dynamic mpich libraries are tried to be linked statically (supposed that this is a bug in cmake 3.6). Possible solution is to use newer cmake.

Workaround: build fdtd3d binary manually by replacing paths to dynamic libs with paths to static libs or add `-L` before `.so`.

Libs trying to be linked statically:
```sh
/bgsys/drivers/V1R4M2_200_2010-100508P/ppc/comm/default/lib/libcxxmpich.cnk.so
/bgsys/drivers/V1R4M2_200_2010-100508P/ppc/comm/default/lib/libmpich.cnk.so
/bgsys/drivers/V1R4M2_200_2010-100508P/ppc/gnu-linux/powerpc-bgp-linux/lib/librt.so
```

Updated command:
```sh
/bgsys/drivers/ppcfloor/comm/bin/mpicxx    -D_FORCE_INLINES -qmaxmem=-1  -O3  -O3 -DNDEBUG   CMakeFiles/fdtd3d.dir/main.cpp.o  -o fdtd3d /bgsys/drivers/V1R4M2_200_2010-100508P/ppc/comm/default/lib/libcxxmpich.cnk.a /bgsys/drivers/V1R4M2_200_2010-100508P/ppc/comm/default/lib/libmpich.cnk.a /bgsys/drivers/V1R4M2_200_2010-100508P/ppc/comm/default/lib/libopa.a -ldcmf.cnk -ldcmfcoll.cnk -lpthread -lSPI.cna /bgsys/drivers/V1R4M2_200_2010-100508P/ppc/gnu-linux/powerpc-bgp-linux/lib/librt.a Kernels/libKernels.a Settings/libSettings.a Coordinate/libCoordinate.a Grid/libGrid.a Layout/libLayout.a File-Management/Loader/libLoader.a File-Management/Dumper/libDumper.a Scheme/libInternalScheme.a Scheme/libScheme.a File-Management/Loader/libLoader.a File-Management/Dumper/libDumper.a File-Management/libFM.a ../Third-party/EasyBMP/libEasyBMP.a Scheme/libInternalScheme.a Grid/libGrid.a Settings/libSettings.a Helpers/libHelpers.a Layout/libLayout.a Kernels/libKernels.a Coordinate/libCoordinate.a
```

# Build for Tesla CMC

Add this to `cmake` command:
```
-DLINK_NUMA=ON
```

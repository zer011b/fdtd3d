# Build for BlueGene\P

BlueGene\P doesn't have cmake provided, so it should be built from source. Download cmake source and then do the following

```sh
./bootstrap
make

# make install won't work because of lack of root access for ordinary users
```

fdtd3d should also be built in bin folder of cmake (why?).

GCC provided for BlueGene\P is heavily outdated (4.1.2), thus, it doesn't support c++11 features and fdtd3d should be built with -DCXX11_ENABLED=OFF.

## Issues

### IBM XL compilers undefined reference

mpixlcxx and mpixlcxx_r compilers lead to undefined references during linking (supposed that this is a bug in IBM XL compilers). GCC should be used instead (`/bgsys/drivers/ppcfloor/comm/bin/mpicxx`):

```sh
../../../../bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DVALUE_TYPE=d -DCOMPLEX_FIELD_VALUES=ON -DSOLVER_DIM_MODES=DIM3 -DPARALLEL_GRID_DIMENSION=3 -DPRINT_MESSAGE=OFF -DPARALLEL_GRID=ON -DPARALLEL_BUFFER_DIMENSION=x -DCXX11_ENABLED=OFF -DCUDA_ENABLED=OFF -DCUDA_ARCH_SM_TYPE=sm_50 -DDYNAMIC_GRID=OFF -DCOMBINED_SENDRECV=ON -DMPI_CLOCK=OFF -DCMAKE_C_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicc -DCMAKE_CXX_COMPILER=/bgsys/drivers/ppcfloor/comm/bin/mpicxx
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

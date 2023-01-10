# Test suite

Test suite contains different electromagnetics tests, which should **all be successful** in order to merge change into `master`.

## Run on CPU

To run test suite execute next script from home folder of fdtd3d:
```
./Tests/run-testsuite.sh
```

This script will fail on the first failed test. To run each individual test execute next script from home folder of fdtd3d (example for test `t1.1`):
```
./Tests/run-test.sh t1.1 1 0 `pwd`/Tests `pwd`
```

To run all tests, showing info on failed and successful tests, execute this:
```
for i in t1.1 t1.2 t2.1 t2.2 t2.3 t3 t4.1 t4.2 t4.3 t5 t6.1 t6.2 t6.3 t6.4 t6.5 t6.6 t6.7 t6.8 t6.9 t6.10 t6.11 t6.12 t6.13 t7.1 t7.2 t7.3 t7.4 t7.5 t7.6 t8; do
  ./Tests/run-test.sh $i 1 0 `pwd`/Tests `pwd`
done
```

To run test suite with specific compiler execute next script from home folder of fdtd3d:
```
./Tests/run-testsuite.sh 1 0 $C_COMPILER $CXX_COMPILER
```

## Run on GPU

To run test suite execute next script from home folder of fdtd3d:
```
./Tests/run-testsuite.sh 1 1
```

To run each individual test on GPU execute next script from home folder of fdtd3d (example for test `t1.1`):
```
./Tests/run-test.sh t1.1 1 1 `pwd`/Tests `pwd`
```

Do not forget to use `cmake >= 3.8` for Cuda builds.

## Run under QEMU in rootfs

To run test suite execute next script from home folder of fdtd3d (example for arm64):
```
export ROOTFS=`pwd`/rootfs/arm64
sudo ./create-ubuntu-rootfs.sh arm64 jammy
TOOLCHAIN_VER=$(ls $ROOTFS/usr/include/c++/)
sed -i "s,set(TOOLCHAIN_VERSION \"7\"),set(TOOLCHAIN_VERSION \"$TOOLCHAIN_VER\")," arm64-gcc-toolchain.cmake
./Tests/run-testsuite.sh 0 0 aarch64-linux-gnu-gcc aarch64-linux-gnu-g++ arm64-gcc-toolchain.cmake
```

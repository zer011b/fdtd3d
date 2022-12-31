# I. CI Testing

Testing for pull requests is performed using github actions CI: open pull request with your changes and it will be automatically tested. For details, see [build.yml](../.github/workflows/build.yml), [unit-test.yml](../.github/workflows/unit-test.yml) and [test-suite.yml](../.github/workflows/test-suite.yml). Cuda tests are only built but not launched in github actions CI, in other cases github actions CI is enough for release testing. For arm, arm64, riscv64, ppc64el testing is done under qemu.

# II. Manual Testing

fdtd3d tests consists of three parts:
- tests of build
- unit tests
- functional tests from test suite

## 1. Tests of build

`./Tools/test-build.sh` script is used to test builds with different flags.

*Note: some combinations like Cuda debug builds for all solver dim modes might take hours to complete, thus, they are explicitly forbidden in cmake config.*

## 2. Unit tests

- `./Tools/test-units.sh` script is used to run unit tests

- `./Tools/test-units-mpi.sh` script is used to run unit tests with mpi

- `./Tools/test-units-cuda.sh` script is used to run unit tests for cuda

## 3. Test suite

`./Tests/run-testsuite.sh` script is used to run tests from test suite.

# III. Release testing

CI for master branch handles common build and tests combinations, however, it doesn't cover all available build options, architectures and test launches. Yet, difference is not that critical (mostly in solver dimensions modes, plus long double floating point precision), so, manual testing before release is needed only for cuda (unit tests and test suite launch). For all test scenarios only GCC compiler is used, but versions may vary.

**Currently supported CPU architectures:**
- x64
- aarch32(armhf)
- aarch64
- riscv64
- ppc64el

**Currently supported platforms:**
- x64, sequential
- x64, with mpi
- x64, with cuda
- x64, with cuda and mpi
- aarch32(armhf), sequential
- aarch64, sequential
- riscv64, sequential
- ppc64el, sequential

## 1. Tests of build

For tests of build this means that next command should be finished successfully for each architecture:
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "" "" ""
```

However, even this might take too much time. Only next combinations are tested for release:

#### x64, sequential, mpi and cuda testing:
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" ""
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "ON,3,xyz" "OFF,sm" "ALL" ""
./Tools/test-build.sh <home_dir> <build_dir> "" "Release Debug" "ON" "" "" "f d" "OFF,1,x" "ON,sm_35" "ALL" ""
./Tools/test-build.sh <home_dir> <build_dir> "" "Release Debug" "ON" "" "" "f d" "ON,3,xyz" "ON,sm_35" "ALL" ""
```

#### x64, aarch32(armhf) cross build, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "arm-linux-gnueabihf-gcc,arm-linux-gnueabihf-g++" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" "arm-gcc-toolchain.cmake"
```

#### x64, aarch64 cross build, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "aarch64-linux-gnu-gcc,aarch64-linux-gnu-g++" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" "arm64-gcc-toolchain.cmake"
```

#### x64, riscv64 cross build, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "riscv64-linux-gnu-gcc,riscv64-linux-gnu-g++" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" "riscv64-gcc-toolchain.cmake"
```

#### x64, ppc64el cross build, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "powerpc64le-linux-gnu-gcc,powerpc64le-linux-gnu-g++" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" "ppc64el-gcc-toolchain.cmake"
```

#### OPTIONAL: aarch32(armhf), sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" ""
```

#### OPTIONAL: aarch64, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" ""
```

#### OPTIONAL: riscv64, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" ""
```

#### OPTIONAL: ppc64el, sequential
```sh
./Tools/test-build.sh <home_dir> <build_dir> "" "" "" "" "" "f d" "OFF,1,x" "OFF,sm" "ALL" ""
```

## 2. Unit tests

Next command should be finished successfully for each architecture:
```sh
 ./Tools/test-units.sh <home_dir> <build_dir> "" "" "" "" "" "" ""
```

Next command should be finished successfully for x64 mpi architecture:
```sh
 ./Tools/test-units-mpi.sh <home_dir> <build_dir> "" "mpicc,mpicxx" "" "" "" ""
```

Next command should be finished successfully for x64 cuda architecture:
```sh
./Tools/test-units-cuda.sh <home_dir> <build_dir> "" "" "RelWithDebInfo" "ON" "" "f d" "" ""
```

## 3. Test suite

Next command should be finished successfully for each architecture:
```sh
./Tests/run-testsuite.sh 0 0
```

Next command should be finished successfully for x64 mpi architecture:
```sh
./Tests/run-testsuite.sh 1 0 mpicc mpicxx
```

Next command should be finished successfully for x64 cuda architecture:
```sh
./Tests/run-testsuite.sh 0 1
```

Next command should be finished successfully for x64 cuda/mpi architecture:
```sh
./Tests/run-testsuite.sh 1 1 mpicc mpicxx
```

# CI Testing

Testing for pull requests is performed using Travis CI: open pull request with your changes and it will be automatically tested. For details, see [.travis.yml](../.travis.yml). Cuda tests are only built but not launched on Travis CI.

# Manual Testing

fdtd3d tests consists of three parts:
- tests of build
- unit tests
- functional tests from test suite

## Tests of build

`./Tools/test-build.sh` script is used to test builds with different flags:

```
# To test builds of all combinations:
# ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "" "" ""
#
# To test builds of all combinations except for cuda:
# ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "OFF,sm" "" ""
#
# To test builds of all sequential combinations:
# ./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" ""
```

*Note: some combinations like Cuda debug builds for all solver dim modes might take hours to complete, thus, they are forbidden.*

# Release testing

CI for master branch handles common build and tests combinations, however, it doesn't cover all available build options, architectures and test launches. So, before release, full testing is performed for all supported platforms. For all test scenarios only GCC compiler is used.

**Currently supported CPU architectures:**
- x64
- aarch32(armhf)
- aarch64

**Currently supported platforms:**
- x64, sequential
- x64, with mpi
- x64, with cuda
- x64, with cuda and mpi
- aarch32(armhf), sequential
- aarch64, sequential

## Tests of build

For tests of build this means that next command should be finished successfully for each architecture:
```sh
./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "" "" "" ""
```

However, even this might take too much time. Only next combinations are tested for release:

#### x64, sequential, mpi and cuda testing:
```sh
./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" ""
./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "ON,3,xyz" "OFF,sm" "" ""
./test-build.sh <home_dir> <build_dir> "" "Release Debug" "" "" "" "" "OFF,1,x" "ON,sm_35" "" ""
./test-build.sh <home_dir> <build_dir> "" "Release Debug" "" "" "" "" "ON,3,xyz" "ON,sm_35" "" ""
```

#### x64, aarch32(armhf) cross build, sequential
```sh
./test-build.sh <home_dir> <build_dir> "arm-linux-gnueabihf-gcc,arm-linux-gnueabihf-g++" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" "arm-gcc-toolchain.cmake"
```

#### x64, aarch64 cross build, sequential
```sh
./test-build.sh <home_dir> <build_dir> "aarch64-linux-gnu-gcc,aarch64-linux-gnu-g++" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" "arm64-gcc-toolchain.cmake"
```

#### aarch32(armhf), sequential
```sh
./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" ""
```

#### aarch64, sequential
```sh
./test-build.sh <home_dir> <build_dir> "" "" "" "" "" "" "OFF,1,x" "OFF,sm" "" ""
```

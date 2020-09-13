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

*Note: some combinations like Cuda debug builds for all solver dim modes might take hours to complete, try to avoid them.*

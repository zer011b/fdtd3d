# Test suite

Test suite contains different electromagnetics tests, which should **all be successful** in order to merge change into `master`.

# Run on CPU

To run test suite execute next script from home folder of fdtd3d:
```
./run-testsuite.sh
```

This script wil fail on the first failed test. To run each individual test execute next script from home folder of fdtd3d (example for test `t1.1`):
```
./Tests/run-test.sh t1.1 `pwd`/Tests `pwd`
```

To run all tests, showing info on failed and successful tests, execute this:
```
for i in t1.1 t1.2 t2.1 t2.2 t2.3 t3 t4.1 t4.2 t4.3 t5 t6.1 t6.2 t6.3 t6.4 t6.5 t6.6 t6.7 t6.8 t6.9 t6.10 t6.11 t6.12 t6.13 t7.1 t7.2 t7.3 t7.4 t7.5 t7.6 t8; do
  ./Tests/run-test.sh $i `pwd`/Tests `pwd`
done
```

# Run on GPU

To run each individual test on GPU execute next script from home folder of fdtd3d (example for test `t1.1`):
```
./Tests/run-test-gpu.sh t1.1 `pwd`/Tests `pwd`
```

Do not forget to use `cmake >= 3.8` for Cuda builds.

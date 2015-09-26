# fdtd3d
3D FDTD solver.

# Build process
Build is done using cmake & make.
```
mkdir build
cd build
cmake ..
make
```

# About
This solver is an MPI/OpenMP implementation of fdtd solver. Solver currently supports only 3D grids. Solver is manually set to perform calculations of one specific problem and needs to be manually modified to perform other calculations. Current code structure is not the best and will be modified in future. Performance will be tested and improved as well.

EasyBMP lib is used to output resulting electromagnetic fields. It is downloaded from sourceforge and used as is.

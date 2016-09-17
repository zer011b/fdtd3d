# Build for BlueGene\P

BlueGene\P doesn't have cmake provided, so it should be built from source. Download cmake source and then do the following

```sh
./bootstrap
make

# make install won't work because of lack of root access for ordinary users
```

fdtd3d should also be built in bin folder of cmake (why?).

GCC provided for BlueGene\P is heavily outdated (4.1.2), thus, it doesn't support c++11 features and fdtd3d should be built with -DCXX11_ENABLED=OFF.

add_executable (fdtd3d-convert converter.cpp)
target_link_libraries (fdtd3d-convert Dumper Loader Settings Grid)

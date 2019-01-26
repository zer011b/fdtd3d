message ("Converter build.")

add_executable (fdtd3d-convert converter.cpp)

if ("${CXX11_ENABLED}")
  target_compile_features (fdtd3d-convert PUBLIC cxx_std_11)
endif ()

target_link_libraries (fdtd3d-convert Dumper Loader Settings Grid)

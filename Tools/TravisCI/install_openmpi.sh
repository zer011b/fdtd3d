#!/bin/bash

set -e

# for macOS builds use OpenMPI from homebrew
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    cd openmpi
    # check to see if OpenMPI is cached from previous build
    if [ -f "bin/mpirun" ]; then
	echo "Using cached OpenMPI"
    else
        echo "Installing OpenMPI with homebrew"
	HOMEBREW_TEMP=$TRAVIS_BUILD_DIR/openmpi
        brew install open-mpi
    fi
else
    # for Ubuntu builds install OpenMPI from source
    # check to see if OpenMPI is cached from previous build
    if [ -f "openmpi/bin/mpirun" ] && [ -f "openmpi-2.0.1/config.log" ]; then
	echo "Using cached OpenMPI"
	echo "Configuring OpenMPI"
	cd openmpi-2.0.1
	./configure --prefix=$TRAVIS_BUILD_DIR/openmpi CC=$C_COMPILER CXX=$CXX_COMPILER &> openmpi.configure
    else
        # install OpenMPI from source
	echo "Downloading OpenMPI Source"
	wget https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.1.tar.gz
	tar zxf openmpi-2.0.1.tar.gz
	echo "Configuring and building OpenMPI"
	cd openmpi-2.0.1
	./configure --prefix=$TRAVIS_BUILD_DIR/openmpi CC=$C_COMPILER CXX=$CXX_COMPILER &> openmpi.configure
	make -j4 &> openmpi.make
	make install &> openmpi.install
	cd ..
    fi
    # recommended by Travis CI documentation to unset these for MPI builds
    test -n $CC && unset CC
    test -n $CXX && unset CXX
fi

#!/bin/bash

#
#  Copyright (C) 2022 Gleb Balykov
#
#  This file is part of fdtd3d.
#
#  fdtd3d is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  fdtd3d is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with fdtd3d; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# Create Ubuntu rootfs, see https://wiki.ubuntu.com/ARM/RootfsFromScratch/QemuDebootstrap for more details
#
# ./create-ubuntu-rootfs <arch> <version> <rootfs_dir>
# or
# ./create-ubuntu-rootfs <arch> <version>

set -e

# Architecture of rootfs
ARCH=$1; shift
if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "armhf" ] && [ "$ARCH" != "riscv64" ]; then
  echo "Only next architectures are supported: arm64, armhf, riscv64."
  exit 1
fi

# Version
VERSION=$1; shift
if [ "$VERSION" != "trusty" ] && [ "$VERSION" != "xenial" ] && [ "$VERSION" != "bionic" ] && [ "$VERSION" != "focal" ] && [ "$VERSION" != "jammy" ]; then
  echo "Only next ubuntu versions are supported: 14.04 (trusty), 16.04 (xenial), 18.04 (bionic), 20.04 (focal), 22.04 (jammy)"
  exit 1
fi

if [ "$ARCH" == "riscv64" ]; then
  if [ "$VERSION" == "trusty" ] || [ "$VERSION" == "xenial" ] || [ "$VERSION" == "bionic" ]; then
    echo "riscv64 is supported since ubuntu 20.04 (focal)"
    exit 1
  fi
fi

# Rootfs directory (can be empty)
ROOTFS=$1;

if [ "$ROOTFS" == "" ]; then
  ROOTFS=`pwd`/rootfs/$ARCH
  mkdir -p $ROOTFS
fi

PACKAGES=build-essential,symlinks
URL=http://ports.ubuntu.com

qemu-debootstrap --arch $ARCH --components main,universe --include $PACKAGES $VERSION $ROOTFS $URL
chroot $ROOTFS symlinks -cr /usr

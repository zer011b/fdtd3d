#!/bin/bash

#
#  Copyright (C) 2025 Gleb Balykov
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

# Create Debian rootfs
#
# ./create-debian-rootfs <arch> <version> <rootfs_dir>
# or
# ./create-debian-rootfs <arch> <version>

set -e

# Architecture of rootfs
ARCH=$1; shift
if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "armhf" ] && [ "$ARCH" != "ppc64el" ] && [ "$ARCH" != "riscv64" ] && [ "$ARCH" != "loongarch64" ]; then
  echo "Only next architectures are supported: arm64, armhf, ppc64el, riscv64, loongarch64 (loong64)."
  exit 1
fi

if [ "$ARCH" == "loongarch64" ]; then
  ARCH="loong64"
fi

# Version
VERSION=$1; shift
if [ "$VERSION" != "sid" ]; then
  echo "Only next debian versions are supported: sid."
  exit 1
fi

# Rootfs directory (can be empty)
ROOTFS=$1;

if [ "$ROOTFS" == "" ]; then
  if [ "$ARCH" == "loong64" ]; then
    ROOTFS=`pwd`/rootfs/loongarch64
  else
    ROOTFS=`pwd`/rootfs/$ARCH
  fi
  mkdir -p $ROOTFS
fi

PACKAGES=""
URL=http://ftp.debian.org/debian/

qemu-debootstrap --foreign --arch $ARCH --components main,universe $VERSION $ROOTFS $URL
#chroot $ROOTFS symlinks -cr /usr

# Cross build for Raspberry Pi (arm32)

Download Raspbian image file (https://www.raspberrypi.org/downloads/raspbian/) and create rootfs:

```sh
mkdir /mnt/rpi
mount -o ro,loop,offset=<offset> -t auto <path_to_image> /mnt/rpi
```

To get offset run:
```sh
fdisk -l <path_to_image> | grep "Linux" | awk '{print $2 * 512}'
```

Make copies:
```sh
sudo chroot $ROOTFS
cp /usr/lib/gcc/arm-linux-gnueabihf/8/crtendS.o /usr/lib/arm-linux-gnueabihf/
cp /usr/lib/gcc/arm-linux-gnueabihf/8/crtbeginS.o /usr/lib/arm-linux-gnueabihf/
```

Then pass `-DCMAKE_TOOLCHAIN_FILE=<path_to_arm-gcc-toolchain.cmake>`.

By default, Raspberry Pi cmake toolchain file is configured to use rootfs files from `/mnt/rpi`. Note that rootfs should match image installed on your Raspberry Pi.

To check that correct libs from rootfs have been linked, add this to toolchain file:
```cmake
add_link_options("-Wl,--verbose")
```

# Cross build for Raspberry Pi (arm64)

Raspbian is 32bit, download Ubuntu Server instead (https://ubuntu.com/download/raspberry-pi). Not all required libs are preinstalled in image, so install them manually.

Prepare rootfs:
```sh
sudo mount -t proc /proc $ROOTFS/proc
sudo mount -t sysfs /sys $ROOTFS/sys
sudo mount -o bind /dev $ROOTFS/dev
sudo mount -o bind /dev/pts $ROOTFS/dev/pts
sudo cp /etc/resolv.conf $ROOTFS/etc/resolv.conf
```

Install libs:
```sh
sudo chroot $ROOTFS
apt update
apt install gcc g++ build-essential
```

Symlinks are absolute which leads to unresolved paths and strange errors like:
```
/usr/lib/gcc-cross/aarch64-linux-gnu/9/../../../../aarch64-linux-gnu/bin/ld: /mnt/rpi/usr/lib/aarch64-linux-gnu//libm.a(s_sin.o): relocation R_AARCH64_ADR_PREL_PG_HI21 against symbol `__stack_chk_guard@@GLIBC_2.17' which may bind externally can not be used when making a shared object; recompile with -fPIC
/usr/lib/gcc-cross/aarch64-linux-gnu/9/../../../../aarch64-linux-gnu/bin/ld: /mnt/rpi/usr/lib/aarch64-linux-gnu//libm.a(s_sin.o)(.text+0xc): unresolvable R_AARCH64_ADR_PREL_PG_HI21 relocation against symbol `__stack_chk_guard@@GLIBC_2.17'
```

Make copies instead of symlinks:
```sh
sudo chroot $ROOTFS
rm /usr/lib/gcc/aarch64-linux-gnu/7/libgcc_s.so
cp /lib/aarch64-linux-gnu/libgcc_s.so.1 /usr/lib/gcc/aarch64-linux-gnu/7/libgcc_s.so
rm /usr/lib/aarch64-linux-gnu/libm.so
cp /lib/aarch64-linux-gnu/libm.so.6 /usr/lib/aarch64-linux-gnu/libm.so
```

**Other symlinks might be broken too, but this doesn't affect build of fdtd3d.**

Make copies:
```sh
sudo chroot $ROOTFS
cp /usr/lib/gcc/aarch64-linux-gnu/7/crtendS.o /usr/lib/aarch64-linux-gnu/
cp /usr/lib/gcc/aarch64-linux-gnu/7/crtbeginS.o /usr/lib/aarch64-linux-gnu/
```

Everything else is same as for arm32, except that `arm64-gcc-toolchain.cmake` should be used.

### Alternative: debotstrap ubuntu rootfs

Another way is to create rootfs with all required libs from scratch: https://wiki.ubuntu.com/ARM/RootfsFromScratch/QemuDebootstrap.

# Troubleshooting

## Non-standard system paths

For systems with non-standard paths (like `/lib64/`) `-L<path>` option might not be enough to find libararies, since `-L` specifies search dirs only for explicitly specified libs (i.e. through `-l<libname>` in command line).

Dependencies of these libs are searched in default paths. To add custom paths for dependencies search add `-Wl,--rpath-link=<path>`. Option `--rpath-link` is enough and `--rpath` is not required since linker on the target system should know about its system paths already. For more details, see: https://linux.die.net/man/1/ld.

If cross toolchain, which targets specific system, is used, then there should be no problem with non-standard paths.

## Broken symlinks

Sometimes symlinks get broken in rootfs. For example, next error message probably means that symlinks are broken:
```
warning: Using 'dlopen' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking
/mnt/rpi/usr/lib/arm-linux-gnueabihf/libdl.a(dlopen.o): In function `dlopen':
(.text+0xc): undefined reference to `__dlopen'
```

This can be then verified by addition of `-Wl,--verbose` option to linker options, it will show:
```
attempt to open /mnt/rpi/usr/lib/arm-linux-gnueabihf/libdl.so failed
```

And for final verification run `file /mnt/rpi/usr/lib/arm-linux-gnueabihf/libdl.so`:
```
/mnt/rpi/usr/lib/arm-linux-gnueabihf/libdl.so: broken symbolic link to /lib/arm-linux-gnueabihf/libdl.so.2
```

To fix all such symlinks run (on rootfs mounted with write access):
```sh
for file in `find /mnt/rpi -name "*.so"`; do
  target=$(file $file | grep broken | awk '{print $6}')
  if [ "$target" != "" ]; then
    sudo rm $file
    sudo ln -s /mnt/rpi/$target $file
  fi
done
```

# Custom toolchain

There's always a solution to build/use a custom toolchain for your specific target (e.g. custom toolchain for Raspberry Pi). This will solve most of the problems related to passing correct compiler options, because correct rootfs and paths will already be incorporated in such toolchain.

# Manual invocation of cross compiler

For reference, to compile simple main.cpp using custom rootfs next command can be used.

Clang:
```sh
clang++ --target=aarch64-linux-gnu -isystem <rootfs/ipath> --sysroot=<rootfs> --gcc-toolchain=<rootfs/toolchainpath> -B<rootfs/crtbeginpath> -L<rootfs/linkpaths> -Wl,--rpath-link=<rootfs/linkpaths> main.cpp
```

GCC:
```sh
aarch64-linux-gnu-g++ -isystem <rootfs/ipath> --sysroot=<rootfs> -B<rootfs/crt1path> -L<rootfs/linkpaths> -Wl,--rpath-link=<rootfs/linkpaths> main.cpp
```

For standard rootfs layout this should work fine. Yet, for both GCC and Clang there might be problems with linker because of `-Bprefix` option, if it is specified incorrectly for some non-standard rootfs.
```
GCC -> /lib/ld-linux.so: No such file or directory.
Clang -> /usr/bin/aarch64-linux-gnu-ld: cannot find crtbegin.o: No such file or directory
```

So, be sure to specify `-Bprefix` correctly, for example for GCC it should be `/usr/lib` or `/usr/lib64` (i.e. path where `crt1.o` is located), and for Clang it should point to location of `crtbegin.o`. If `-Bprefix` is not used, default compiler libraries, include and data files (e.g. `crt1.o`) will be found instead of the ones from rootfs. To check that correct files are used `-Wl,--verbose` option can be used.

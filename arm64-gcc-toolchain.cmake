set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(TOOLCHAIN "aarch64-linux-gnu")
set(TOOLCHAIN_VERSION "7")
set(CROSS_ROOTFS "/mnt/rpi")

set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

include_directories(BEFORE SYSTEM ${CROSS_ROOTFS}/usr/include/c++/${TOOLCHAIN_VERSION})
include_directories(BEFORE SYSTEM ${CROSS_ROOTFS}/usr/include/${TOOLCHAIN}/c++/${TOOLCHAIN_VERSION})

set(CMAKE_SYSROOT "${CROSS_ROOTFS}")

#add_link_options("-Wl,--verbose")
add_link_options("-B${CROSS_ROOTFS}/usr/lib/${TOOLCHAIN}")
add_link_options("-L${CROSS_ROOTFS}/lib")
add_link_options("-L${CROSS_ROOTFS}/usr/lib")
add_link_options("-L${CROSS_ROOTFS}/usr/lib/gcc/${TOOLCHAIN}/${TOOLCHAIN_VERSION}")
add_link_options("-L${CROSS_ROOTFS}/usr/lib/${TOOLCHAIN}/")
add_link_options("-L${CROSS_ROOTFS}/lib/${TOOLCHAIN}/")

# Architecture
add_compile_options(-march=armv8-a+fp+crc+simd)
# Microarchitecture
add_compile_options(-mtune=cortex-a72)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#pragma once

// Minimal stddef.h shim for NVRTC device compilation.
// We avoid pulling in host libc headers (glibc) which are not device-annotated.

typedef unsigned long long size_t;
typedef long long ptrdiff_t;

#ifndef NULL
#define NULL 0
#endif


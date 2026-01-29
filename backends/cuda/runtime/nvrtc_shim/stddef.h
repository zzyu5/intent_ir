#pragma once

// Minimal stddef.h shim for NVRTC device compilation.
//
// NVRTC already provides `size_t` / `ptrdiff_t` via its builtin header. We only
// define `NULL` to satisfy legacy code paths that still include <stddef.h>.

#ifndef NULL
#define NULL 0
#endif

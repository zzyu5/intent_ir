#pragma once

// Minimal math.h shim for NVRTC device compilation.
// Provide INFINITY/NAN and route math intrinsics to CUDA's device headers.

#include <math_constants.h>
#include <math_functions.h>

#ifndef INFINITY
#define INFINITY CUDART_INF_F
#endif

#ifndef NAN
#define NAN CUDART_NAN_F
#endif


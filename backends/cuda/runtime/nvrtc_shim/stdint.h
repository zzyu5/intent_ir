#pragma once

// Minimal stdint.h shim for NVRTC device compilation.
// Define the fixed-width integer types used by IntentIR CUDA runtime headers.

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

typedef long long intptr_t;
typedef unsigned long long uintptr_t;


#pragma once
// NVRTC compatibility shim.
//
// Some glibc header layouts (notably in minimal/driver-only environments) may
// reference `gnu/stubs-32.h` even when 32-bit development headers are not
// installed. NVRTC needs the include to succeed for compilation, even though
// the device code we compile does not rely on 32-bit ABI stubs.
//
// This header is intentionally empty.


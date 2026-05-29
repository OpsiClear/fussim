#pragma once
// Windows compatibility shim — MUST be included before any PyTorch/CUDA header.
//
// The Windows SDK header rpcndr.h does `#define small char`. It is pulled in
// transitively by PyTorch/CUDA headers (e.g. c10's CUDACachingAllocator, whose
// StreamSegmentSize takes a `bool small` parameter), which then fails to compile
// with "invalid combination of type specifiers" because `small` expands to `char`.
//
// A bare `#ifdef small / #undef small` placed before the torch includes is a no-op:
// at that point no Windows header has run yet, so `small` is not defined. Instead we
// pull the Windows headers in here first, then undef the macro, so by the time the
// torch headers below are parsed the macro is gone. windows.h has an include guard,
// so torch's later (re)inclusion does not redefine it.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef small
#undef small
#endif
#endif

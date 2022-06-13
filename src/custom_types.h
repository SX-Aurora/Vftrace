#ifndef CUSTOM_TYPES_H
#define CUSTOM_TYPES_H

#include "config.h"
#include <stdint.h>

// in the unlikely event that the c-compiler does not support the uintptr_t type
// define it to match a unsigned int of same size as a void pointer
#ifndef _HAS_UINTPTR
#if SIZEOF_VOID_P == SIZEOF_UINT8_T
typedef uint8_t uintptr_t;
#elif SIZEOF_VOID_P == SIZEOF_UINT16_T
typedef uint16_t uintptr_t;
#elif SIZEOF_VOID_P == SIZEOF_UINT32_T
typedef uint32_t uintptr_t;
#else
typedef uint64_t uintptr_t;
#endif
#endif

// create an integer type that is large enough to hold a float
// for radix sorting
#if SIZEOF_FLOAT == SIZEOF_UINT8_T
typedef uint8_t uintflt_t;
#elif SIZEOF_FLOAT == SIZEOF_UINT16_T
typedef uint16_t uintflt_t;
#elif SIZEOF_FLOAT == SIZEOF_UINT32_T
typedef uint32_t uintflt_t;
#else
typedef uint64_t uintflt_t;
#endif

#if SIZEOF_DOUBLE == SIZEOF_UINT8_T
typedef uint8_t uintdbl_t;
#elif SIZEOF_DOUBLE == SIZEOF_UINT16_T
typedef uint16_t uintdbl_t;
#elif SIZEOF_DOUBLE == SIZEOF_UINT32_T
typedef uint32_t uintdbl_t;
#else
typedef uint64_t uintdbl_t;
#endif

#endif

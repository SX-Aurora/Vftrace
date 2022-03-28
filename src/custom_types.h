#ifndef CUSTOM_TYPES_H
#define CUSTOM_TYPES_H

#include "config.h"

// in the unlikely event that the c-compiler does not support the uintptr_t type
// default it to unsigned long long
#ifndef _HAS_UINTPTR
typedef unsigned long long int uintptr_t;
#else
#include <stdint.h>
#endif

// create an integer type that is large enough to hold a double
// for radix sorting
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

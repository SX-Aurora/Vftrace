#ifndef ADDRESS_TYPE_H
#define ADDRESS_TYPE_H

// in the unlikely event that the c-compiler does not support the uintptr_t type
// default it to unsigned long long
#ifndef _HAS_UINTPTR
typedef unsigned long long int uintptr_t;
#endif

#endif

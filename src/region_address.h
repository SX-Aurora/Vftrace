#ifndef REGION_ADDRESS_H
#define REGION_ADDRESS_H

// Getting the region address is defined here as a macro,
// so it wont mess up the adresses by changing the function stack
#ifdef __ve__
#define GET_REGION_ADDRESS(ADDR) \
   do { \
      asm volatile ("or %0,0,%%lr" : "=r" (ADDR)); \
   } while (0)
#else
#define GET_REGION_ADDRESS(ADDR) \
   do { \
      asm volatile ("mov 8(%%rbp), %0" : "=r" (ADDR)); \
   } while (0)
#endif

#endif

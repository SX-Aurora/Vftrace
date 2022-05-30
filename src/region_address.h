#ifndef REGION_ADDRESS_H
#define REGION_ADDRESS_H

// Getting the region address is defined here as a macro,
// so it wont mess up the adresses by changing the function stack
#define GET_REGION_ADDRESS(ADDR) \
   do { \
      (ADDR) = (__builtin_return_address(0)); \
   } while (0)

#endif

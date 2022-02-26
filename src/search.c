#include <stdint.h>

#include "address_type.h"
#include "symbols.h"

int vftr_binary_search_uint64(int n, uint64_t *list, uint64_t value) {
   int low = 0;
   int high = n-1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      if (list[mid] > value) {
         high = mid - 1;
      } else if (list[mid] < value) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

int vftr_binary_search_symboltable(int nsymb, symbol_t *symbols,
                                   uintptr_t address) {
   int low = 0;
   int high = nsymb -1;
   while (low <= high) {
      int mid = (low+high) >> 1;
      if (symbols[mid].addr > address) {
         high = mid - 1;
      } else if (symbols[mid].addr < address) {
         low = mid + 1;
      } else {
         return mid;
      }
   }
   return -1; // not found
}

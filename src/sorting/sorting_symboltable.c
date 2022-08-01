#include <stdlib.h>

#include "custom_types.h"
#include "symbols.h"

// sort the symboltable with a quick-sort
void vftr_sort_symboltable(unsigned int nsymb, symbol_t *symbols) {
   if (nsymb < 2) return;
   uintptr_t pivot = symbols[nsymb/2].addr;
   int left, right;
   for (left=0, right=nsymb-1; ; left++, right--) {
      while (symbols[left].addr < pivot) left++;
      while (symbols[right].addr > pivot) right--;
      if (left >= right) break;
      symbol_t tmp = symbols[left];
      symbols[left] = symbols[right];
      symbols[right] = tmp;
   }

   vftr_sort_symboltable(left, symbols);
   vftr_sort_symboltable(nsymb-left, symbols+left);
}

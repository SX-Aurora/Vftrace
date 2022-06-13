#include <stdlib.h>

#include "custom_types.h"
#include "symbols.h"

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols) {
   // create the fuckets for sorting
   // use one array to store both buckets
   symbol_t *buckets = (symbol_t*) malloc(2*nsymb*sizeof(symbol_t));
   // loop over the bits of the symbols address which is (void*)
   int nbits = 8*sizeof(uintptr_t);
   for (int ibit=0; ibit<nbits; ibit++) {
      int idx[2] = {0,0};
      // sort the symbols into buckets base on thei ibit-th bit
      for (unsigned int i=0; i<nsymb; i++) {
         // extract the ibit-th bit of the i-th symbols address
         int bit = (symbols[i].addr >> ibit) & 1llu;
         // add the symbol to the selected bucket
         buckets[bit*nsymb+idx[bit]] = symbols[i];
         idx[bit]++;
      }
      // copy the presorted symbols back to the original list
      for (int i=0; i<idx[0]; i++) {
         symbols[i] = buckets[0*nsymb+i];
      }
      for (int i=0; i<idx[1]; i++) {
         symbols[idx[0]+i] = buckets[1*nsymb+i];
      }
   }
   free(buckets);
}

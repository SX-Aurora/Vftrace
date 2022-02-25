#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "symbols.h"

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_radixsort_uint64(int n, uint64_t *list) {
   // create the buckets for sorting
   // use one array to store both buckets
   uint64_t *buckets = (uint64_t*) malloc(2*n*sizeof(uint64_t));
   // loop over the bits
   int nbits = 8*sizeof(uint64_t);
   for (int ibit=0; ibit<nbits; ibit++) {
      int idx[2] = {0,0};
      // sort numbers into buckets based on their ibit-th bit
      for (int i=0; i<n; i++) {
         // extract the ibit-th bit of the i-th number
         int bit = (list[i] >> ibit) & 1;
         // add number to the selected bucket
         buckets[bit*n+idx[bit]] = list[i];
         idx[bit]++;
      }
      // copy the presorted numbers back to the original list
      for (int i=0; i<idx[0]; i++) {
         list[i] = buckets[0*n+i];
      }
      for (int i=0; i<idx[1]; i++) {
         list[idx[0]+i] = buckets[1*n+i];
      }
   }
   free(buckets);
}

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols) {
   // create the fuckets for sorting
   // use one array to store both buckets
   symbol_t *buckets = (symbol_t*) malloc(2*nsymb*sizeof(symbol_t));
   // loop over the bits of the symbols address which is (void*)
   int nbits = 8*sizeof(void*);
   for (int ibit=0; ibit<nbits; ibit++) {
      int idx[2] = {0,0};
      // sort the symbols into buckets base on thei ibit-th bit
      for (unsigned int i=0; i<nsymb; i++) {
         // extract the ibit-th bit of the i-th symbols address
         unsigned long long tmpaddr = (unsigned long long) symbols[i].addr;
         int bit = (tmpaddr >> ibit) & 1llu;
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
